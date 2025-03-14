import argparse
import csv
import math
import os
from collections import defaultdict
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from PIL import Image
from safetensors.torch import load_file, save_file
from schedulefree import AdamWScheduleFree
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.transforms import v2
from tqdm import tqdm


def resize_and_crop(image, target_size):
    # Get current dimensions
    width, height = image.size
    target_width, target_height = target_size

    # Calculate aspect ratios
    img_ratio = width / height
    target_ratio = target_width / target_height

    # Determine dimensions for cropping
    if img_ratio > target_ratio:
        # Image is wider than target - crop width
        new_width = int(height * target_ratio)
        offset = (width - new_width) // 2
        image = image.crop((offset, 0, offset + new_width, height))
    else:
        # Image is taller than target - crop height
        new_height = int(width / target_ratio)
        offset = (height - new_height) // 2
        image = image.crop((0, offset, width, offset + new_height))

    # Resize to target dimensions
    image = image.resize(target_size, Image.Resampling.LANCZOS)

    return image


class BucketedAVADataset(Dataset):
    def __init__(
        self,
        csv_path=None,
        ava_txt_path=None,
        transform=None,
        tag_file=None,
        bucket_tolerance=32,
        image_dir=None,
        storage_endpoint="https://ava-boo.us-iad-1.linodeobjects.com/images",
    ):
        self.transform = transform
        self.bucket_tolerance = bucket_tolerance
        self.image_dir = image_dir
        self.storage_endpoint = storage_endpoint

        # Initialize lists
        self.image_ids = []
        self.scores = []
        self.tags = []
        self.image_sizes = []
        self.image_paths = []  # Add this attribute back

        # Parse AVA.txt file if provided
        if ava_txt_path:
            print("Parsing AVA.txt file")
            self._parse_ava_txt(ava_txt_path)

        # Load metadata from CSV if provided
        if csv_path:
            print("Loading metadata from CSV")
            self._load_csv_metadata(csv_path)

        # Load tag descriptions if provided
        self.tag_descriptions = {}
        if tag_file:
            print("Loading tag descriptions")
            self._load_tag_descriptions(tag_file)

        # Pre-compute all image paths
        print("Computing image paths")
        self._compute_image_paths()

        # Create buckets of similar sized images
        print("Creating buckets")
        self.buckets = self._create_buckets()

        print("Mapping buckets")
        # Map from image index to bucket and position within bucket
        self.index_to_bucket = {}
        for bucket_idx, bucket in enumerate(self.buckets):
            for pos_in_bucket, img_idx in enumerate(bucket):
                self.index_to_bucket[img_idx] = (bucket_idx, pos_in_bucket)

        # Calculate target sizes for each bucket
        print("Getting bucket sizes")
        self.bucket_sizes = self._get_bucket_sizes()

    def _parse_ava_txt(self, ava_txt_path):
        """Parse the AVA.txt file for image IDs, scores, and tags."""
        with open(ava_txt_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 15:
                    index = parts[0]
                    img_id = parts[1]
                    # Get counts of ratings 1-10
                    ratings = [int(count) for count in parts[2:12]]
                    # Calculate mean score
                    score = sum((i + 1) * r for i, r in enumerate(ratings)) / sum(
                        ratings
                    )
                    # Get tags
                    semantic_tags = [int(tag) for tag in parts[12:14]]
                    self.image_ids.append(img_id)
                    self.scores.append(score)
                    self.tags.append(semantic_tags)
                    # Initialize with placeholder size, will be updated from CSV
                    self.image_sizes.append((0, 0))

    def _load_csv_metadata(self, csv_path):
        """Load image metadata from CSV file."""
        # Create a mapping of image IDs to sizes
        id_to_size = {}

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_id = row["id"]
                width = int(row["width"])
                height = int(row["height"])
                id_to_size[img_id] = (width, height)

        # If we have existing image IDs from AVA.txt, update their sizes
        if self.image_ids:
            for i, img_id in enumerate(self.image_ids):
                if img_id in id_to_size:
                    self.image_sizes[i] = id_to_size[img_id]
                else:
                    print(f"Warning: No size metadata for image ID {img_id}")
        else:
            # If no AVA.txt was loaded, use the CSV as the primary source
            for img_id, size in id_to_size.items():
                self.image_ids.append(img_id)
                self.image_sizes.append(size)
                # Add placeholder score and tags
                self.scores.append(0.0)
                self.tags.append([])

    def _compute_image_paths(self):
        """Compute paths for all images."""
        self.image_paths = []
        for img_id in self.image_ids:
            path, is_local = self._get_image_path(img_id)
            self.image_paths.append(path)

    def _load_tag_descriptions(self, tag_file):
        """Load tag descriptions from file."""
        with open(tag_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    tag_id = int(parts[0])
                    tag_desc = " ".join(parts[1:])
                    self.tag_descriptions[tag_id] = tag_desc

    def _get_image_path(self, img_id):
        """Get either local path or URL for an image."""
        if self.image_dir:
            # Try local file first
            img_path = os.path.join(self.image_dir, f"{img_id}.jpg")
            if os.path.exists(img_path):
                return img_path, True  # True indicates local file

        # Fall back to object storage URL
        return f"{self.storage_endpoint}/{img_id}.jpg", False  # False indicates URL

    def _create_buckets(self) -> List[List[int]]:
        """Group images with similar dimensions into buckets."""
        # Directly assign images to buckets based on dimensions
        buckets_dict = defaultdict(list)

        for i, (w, h) in enumerate(self.image_sizes):
            if w <= 0 or h <= 0:
                continue

            # Quantize dimensions to create bucket keys
            bucket_w = (w // self.bucket_tolerance) * self.bucket_tolerance
            bucket_h = (h // self.bucket_tolerance) * self.bucket_tolerance

            # Create a key from the quantized dimensions
            bucket_key = (bucket_w, bucket_h)
            buckets_dict[bucket_key].append(i)

        # Convert dictionary to list of lists
        buckets = list(buckets_dict.values())

        # Filter out buckets that are too small (optional)
        buckets = [bucket for bucket in buckets if len(bucket) >= 3]

        return buckets

    def _get_bucket_sizes(self) -> List[Tuple[int, int]]:
        """Calculate the target size for each bucket (median width and height)."""
        bucket_sizes = []

        for bucket in self.buckets:
            if not bucket:
                bucket_sizes.append((224, 224))  # Default size for empty buckets
                continue

            widths = [self.image_sizes[i][0] for i in bucket]
            heights = [self.image_sizes[i][1] for i in bucket]

            # Using median to avoid outliers
            target_w = int(np.median(widths))
            target_h = int(np.median(heights))

            # Ensure dimensions are multiples of bucket_tolerance
            target_w = (target_w // self.bucket_tolerance) * self.bucket_tolerance
            target_h = (target_h // self.bucket_tolerance) * self.bucket_tolerance

            bucket_sizes.append((target_w, target_h))

        return bucket_sizes

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = self.image_paths[idx]

        # Get the bucket for this image and its target size
        if idx in self.index_to_bucket:
            bucket_idx, _ = self.index_to_bucket[idx]
            target_size = self.bucket_sizes[bucket_idx]
        else:
            # Fallback for images not in buckets
            target_size = (224, 224)

        # Determine if path is local or URL
        is_local = self.image_dir and os.path.exists(img_path)

        # Load image from either local file or URL
        if is_local:
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                raise IOError(f"Error loading local image {img_id}: {e}")
        else:
            try:
                response = requests.get(img_path)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert("RGB")
            except Exception as e:
                raise IOError(f"Error loading image {img_id} from {img_path}: {e}")

        # Resize image to the bucket's target size
        image = resize_and_crop(image, target_size)

        if self.transform:
            image = self.transform(image)

        # Get score and tags
        score = self.scores[idx]
        tags = self.tags[idx] if idx < len(self.tags) else []

        return image, score, img_id, tags


class BucketBatchSampler(Sampler):
    """Samples batches according to image buckets."""

    def __init__(self, bucketed_dataset, batch_size: int, shuffle: bool = True):
        self.dataset = bucketed_dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        # Create batches for each bucket
        batches = []

        for bucket in self.dataset.buckets:
            # Shuffle within each bucket if needed
            if self.shuffle:
                bucket_indices = bucket.copy()
                np.random.shuffle(bucket_indices)
            else:
                bucket_indices = bucket

            # Create batches from this bucket
            for i in range(0, len(bucket_indices), self.batch_size):
                batch = bucket_indices[i : i + self.batch_size]
                if len(batch) == self.batch_size:  # Only use full batches
                    batches.append(batch)
                elif (
                    len(batch) > self.batch_size // 2
                ):  # Allow partial batches if they're big enough
                    batches.append(batch)

        # Shuffle the batches if needed
        if self.shuffle:
            np.random.shuffle(batches)

        # Yield batches
        for batch in batches:
            yield batch

    def __len__(self):
        # Approximate number of batches
        return sum(
            (len(bucket) + self.batch_size - 1) // self.batch_size
            for bucket in self.dataset.buckets
        )


def create_bucketed_ava_dataloader(
    image_dir,
    ava_txt_path,
    batch_size=32,
    bucket_tolerance=32,
    shuffle=True,
    transform=None,
    tag_file=None,
):
    """Create a DataLoader with bucketed AVA dataset images."""
    # Create dataset
    dataset = BucketedAVADataset(
        image_dir=image_dir,
        csv_path="images.metadata.csv",
        ava_txt_path=ava_txt_path,
        transform=transform,
        tag_file=tag_file,
        bucket_tolerance=bucket_tolerance,
    )

    # Create batch sampler
    batch_sampler = BucketBatchSampler(dataset, batch_size, shuffle)

    # Create data loader with the batch sampler
    dataloader = DataLoader(
        dataset, batch_sampler=batch_sampler, num_workers=4, pin_memory=True
    )

    return dataloader, dataset


def create_bucketed_ava_dataloader_split(
    image_dir,
    ava_txt_path,
    train_ratio=0.8,
    batch_size=32,
    bucket_tolerance=32,
    shuffle=True,
    transform=None,
    test_transform=None,
    tag_file=None,
    csv_path=None,
    random_seed=42,
):
    """
    Create DataLoaders with bucketed AVA dataset images split into train and test sets.
    Fixed to prevent index out of range errors.
    """
    # Create full dataset first
    full_dataset = BucketedAVADataset(
        image_dir=image_dir,
        csv_path=csv_path,
        ava_txt_path=ava_txt_path,
        transform=None,
        tag_file=tag_file,
        bucket_tolerance=bucket_tolerance,
    )

    # Set random seed for reproducibility
    np.random.seed(random_seed)

    # Get total valid indices
    valid_indices = []
    for bucket in full_dataset.buckets:
        valid_indices.extend(bucket)

    # Shuffle indices for random split
    shuffled_indices = valid_indices.copy()
    np.random.shuffle(shuffled_indices)

    # Split indices
    split_idx = int(len(shuffled_indices) * train_ratio)
    train_indices = shuffled_indices[:split_idx]
    test_indices = shuffled_indices[split_idx:]

    # Create train and test datasets - important: we'll use our own custom dataset classes
    class TrainDataset(Dataset):
        def __init__(self, parent_dataset: BucketedAVADataset, indices, transform=None):
            self.parent = parent_dataset
            self.indices = indices
            self.transform = transform
            # Copy bucket information
            self.bucket_sizes = parent_dataset.bucket_sizes

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            parent_idx = self.indices[idx]
            img_id = self.parent.image_ids[parent_idx]
            img_path = self.parent.image_paths[parent_idx]

            # First determine if it's a local file
            is_local = self.parent.image_dir and os.path.exists(img_path)

            if img_path is None or (is_local and not os.path.exists(img_path)):
                # Fallback in case image wasn't found during initialization
                img_path, is_local = self.parent._get_image_path(img_id)
                if img_path is None:
                    raise FileNotFoundError(f"Cannot find image for ID {img_id}")

            # Get the bucket for this image and its target size
            if parent_idx in self.parent.index_to_bucket:
                bucket_idx, _ = self.parent.index_to_bucket[parent_idx]
                target_size = self.parent.bucket_sizes[bucket_idx]
            else:
                # Fallback for images not in buckets
                target_size = (224, 224)

            # Load the image based on whether it's local or remote
            if is_local:
                try:
                    image = Image.open(img_path).convert("RGB")
                except Exception as e:
                    raise IOError(
                        f"Error loading local image {img_id} from {img_path}: {e}"
                    )
            else:
                try:
                    response = requests.get(img_path)
                    response.raise_for_status()
                    image = Image.open(BytesIO(response.content)).convert("RGB")
                except Exception as e:
                    raise IOError(
                        f"Error loading remote image {img_id} from {img_path}: {e}"
                    )

            image = resize_and_crop(image, target_size)

            if self.transform:
                image = self.transform(image)

            score = self.parent.scores[parent_idx]
            return image, score, img_id

    # Create train and test datasets
    train_dataset = TrainDataset(full_dataset, train_indices, transform)
    test_dataset = TrainDataset(
        full_dataset,
        test_indices,
        test_transform if test_transform is not None else transform,
    )

    # Create bucket samplers for train and test
    train_buckets = defaultdict(list)
    test_buckets = defaultdict(list)

    # Map original indices to new dataset indices
    train_idx_map = {
        orig_idx: new_idx for new_idx, orig_idx in enumerate(train_indices)
    }
    test_idx_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(test_indices)}

    # Organize indices by bucket for train and test - using new indices
    for orig_idx in train_indices:
        if orig_idx in full_dataset.index_to_bucket:
            bucket_idx, _ = full_dataset.index_to_bucket[orig_idx]
            train_buckets[bucket_idx].append(train_idx_map[orig_idx])

    for orig_idx in test_indices:
        if orig_idx in full_dataset.index_to_bucket:
            bucket_idx, _ = full_dataset.index_to_bucket[orig_idx]
            test_buckets[bucket_idx].append(test_idx_map[orig_idx])

    # Convert to list of lists format expected by BucketBatchSampler
    train_buckets_list = [
        indices for _, indices in sorted(train_buckets.items()) if indices
    ]
    test_buckets_list = [
        indices for _, indices in sorted(test_buckets.items()) if indices
    ]

    # Create custom batch samplers
    class SplitBucketBatchSampler(Sampler):
        def __init__(self, buckets, batch_size, shuffle=True):
            self.buckets = buckets
            self.batch_size = batch_size
            self.shuffle = shuffle

        def __iter__(self):
            batches = []

            for bucket in self.buckets:
                if self.shuffle:
                    bucket_indices = bucket.copy()
                    np.random.shuffle(bucket_indices)
                else:
                    bucket_indices = bucket

                for i in range(0, len(bucket_indices), self.batch_size):
                    batch = bucket_indices[i : i + self.batch_size]
                    if len(batch) >= max(1, self.batch_size // 2):
                        batches.append(batch)

            if self.shuffle:
                np.random.shuffle(batches)

            for batch in batches:
                yield batch

        def __len__(self):
            return sum(
                (len(bucket) + self.batch_size - 1) // self.batch_size
                for bucket in self.buckets
            )

    # Create batch samplers
    train_sampler = SplitBucketBatchSampler(train_buckets_list, batch_size, shuffle)
    test_sampler = SplitBucketBatchSampler(test_buckets_list, batch_size, False)

    # Create DataLoaders
    train_dataloader = DataLoader(
        train_dataset, batch_sampler=train_sampler, num_workers=4, pin_memory=True
    )

    test_dataloader = DataLoader(
        test_dataset, batch_sampler=test_sampler, num_workers=4, pin_memory=True
    )

    return train_dataloader, test_dataloader, train_dataset, test_dataset


# class AVADataset(Dataset):
#     def __init__(self, image_dir, ava_txt_path, transform=None, tag_file=None):
#         self.image_dir = image_dir
#         self.transform = transform
#
#         # Parse AVA.txt file
#         self.image_ids = []
#         self.scores = []
#         self.tags = []
#
#         with open(ava_txt_path, "r") as f:
#             for line in f:
#                 parts = line.strip().split()
#                 if len(parts) >= 15:
#                     index = parts[0]
#                     img_id = parts[1]
#                     # Get counts of ratings 1-10
#                     ratings = [int(count) for count in parts[2:12]]
#                     # Calculate mean score
#                     score = sum((i + 1) * r for i, r in enumerate(ratings)) / sum(
#                         ratings
#                     )
#                     # Get tags
#                     semantic_tags = [int(tag) for tag in parts[12:14]]
#
#                     self.image_ids.append(img_id)
#                     self.scores.append(score)
#                     self.tags.append(semantic_tags)
#
#         # Load tag descriptions if provided
#         self.tag_descriptions = {}
#         if tag_file:
#             with open(tag_file, "r") as f:
#                 for line in f:
#                     parts = line.strip().split()
#                     if len(parts) >= 2:
#                         tag_id = int(parts[0])
#                         tag_desc = " ".join(parts[1:])
#                         self.tag_descriptions[tag_id] = tag_desc
#
#     def __len__(self):
#         return len(self.image_ids)
#
#     def __getitem__(self, idx):
#         img_id = self.image_ids[idx]
#         # Assuming images are named as their ID with appropriate extension
#         img_path = None
#         for ext in [".jpg", ".jpeg", ".png"]:
#             test_path = os.path.join(self.image_dir, f"{img_id}{ext}")
#             if os.path.exists(test_path):
#                 img_path = test_path
#                 break
#
#         if img_path is None:
#             # If not found by ID alone, try searching directory
#             for file in os.listdir(self.image_dir):
#                 if file.startswith(img_id):
#                     img_path = os.path.join(self.image_dir, file)
#                     break
#
#         if img_path is None:
#             raise FileNotFoundError(f"Cannot find image for ID {img_id}")
#
#         image = Image.open(img_path).convert("RGB")
#
#         if self.transform:
#             image = self.transform(image)
#
#         score = self.scores[idx]
#         return image, score, img_id


class AestheticPredictor(nn.Module):
    def __init__(self, latent_dim, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # Output between 0 and 1, we'll scale to 1-10
        )

    def forward(self, x):
        return self.model(x) * 9 + 1  # Scale from 0-1 to 1-10


class LatentAestheticsTransformer(nn.Module):
    def __init__(
        self, channels=16, dim_feedforward=1024, nhead=8, num_layers=3, dropout=0.1
    ):
        super(LatentAestheticsTransformer, self).__init__()

        # Dynamic embedding layer to project latents to transformer dimension
        self.embedding_dim = 768  # Standard transformer dimension
        self.embedding = nn.Linear(
            channels, self.embedding_dim
        )  # Initialize embedding layer

        # Position encoding (needed for transformer)
        self.pos_encoder = PositionalEncoding(self.embedding_dim, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Final prediction layers
        self.fc1 = nn.Linear(self.embedding_dim, dim_feedforward // 2)
        self.fc2 = nn.Linear(dim_feedforward // 2, 1)

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        # Initialize embedding layer if not created or if input channels changed
        if self.embedding is None or self.embedding.weight.shape[1] != channels:
            self.embedding = nn.Linear(channels, self.embedding_dim).to(x.device)

        # Reshape latents to sequence format: [B, H*W, C]
        x = x.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)

        # Project to transformer dimension
        x = self.embedding(x)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Pass through transformer
        x = self.transformer_encoder(x)

        # Global pooling (mean across sequence dimension)
        x = torch.mean(x, dim=1)

        # Final prediction
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x.squeeze()


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=7600):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


@torch.no_grad()
def extract_latents(vae_model, images):
    # Use VAE encoder to get latents
    latents = vae_model.encode(images).latent_dist.sample()
    return latents


def train_aesthetic_model(
    train_loader: DataLoader,
    vae_model: AutoencoderKL,
    aesthetic_model: AestheticPredictor,
    num_epochs=10,
    lr=0.001,
):
    # optimizer = torch.optim.Adam(aesthetic_model.parameters(), lr=lr)
    optimizer = AdamWScheduleFree(aesthetic_model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    aesthetic_model.train()
    optimizer.train()

    progress_bar = tqdm(total=len(train_loader))
    for epoch in range(num_epochs):
        running_losses = []

        for images, scores, _ in train_loader:
            # Skip images without scores
            mask = scores >= 0
            if not torch.any(mask):
                continue

            images, scores = (
                images[mask],
                scores[mask].float().view(-1, 1),
            )

            latents = extract_latents(vae_model, images)
            # latents = latents.view(latents.size(0), -1)  # Flatten latents if needed

            optimizer.zero_grad()
            predictions = aesthetic_model(latents)
            loss = criterion(predictions.unsqueeze(-1), scores)
            loss.backward()
            optimizer.step()

            running_losses.append(loss.detach().item())

            progress_bar.update(1)
            progress_bar.set_postfix(
                {"avg_loss": sum(running_losses) / len(running_losses)}
            )

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {sum(running_losses) / len(running_losses)}"
        )

    optimizer.eval()

    return aesthetic_model


def predict_aesthetic_scores(dataloader, vae_model, aesthetic_model):
    results = []

    aesthetic_model.eval()
    with torch.no_grad():
        for images, _, img_names in dataloader:
            images = images

            latents = extract_latents(vae_model, images)
            latents = latents.view(latents.size(0), -1)  # Flatten latents if needed

            scores = aesthetic_model(latents)

            for i, img_name in enumerate(img_names):
                results.append((img_name, scores[i].item()))

    return results


def print_bucket_info(dataset, max_buckets=None):
    """
    Print information about the buckets in a dataset

    Works with both BucketedAVADataset and TrainDataset

    Args:
        dataset: The dataset instance
        max_buckets: Maximum number of buckets to display (None shows all)
    """
    print("Bucket Information:")
    print("===================")

    # Handle different dataset types
    if hasattr(dataset, "buckets"):
        # Direct BucketedAVADataset
        buckets = dataset.buckets
        bucket_sizes = dataset.bucket_sizes if hasattr(dataset, "bucket_sizes") else []
    elif hasattr(dataset, "parent"):
        # TrainDataset (has parent reference)
        # We need to filter buckets to only include indices in this dataset
        parent = dataset.parent
        dataset_indices_set = set(dataset.indices)

        # Rebuild buckets with only indices from this dataset
        buckets = []
        for bucket_idx, original_bucket in enumerate(parent.buckets):
            # Filter to only indices in this dataset
            filtered_bucket = [
                idx for idx in original_bucket if idx in dataset_indices_set
            ]
            if filtered_bucket:
                buckets.append(filtered_bucket)

        bucket_sizes = parent.bucket_sizes if hasattr(parent, "bucket_sizes") else []
    else:
        print("Error: Dataset type not supported for bucket information")
        return

    # Get total number of images
    total_images = sum(len(bucket) for bucket in buckets)
    print(f"Total images: {total_images}")
    print(f"Total buckets: {len(buckets)}")
    print()

    # Sort buckets by size (number of images)
    bucket_info = []
    for i, bucket in enumerate(buckets):
        bucket_size = len(bucket)
        if i < len(bucket_sizes):
            target_w, target_h = bucket_sizes[i]
            bucket_info.append((i, bucket_size, (target_w, target_h)))
        else:
            # Fallback if bucket_sizes is missing
            bucket_info.append((i, bucket_size, (0, 0)))

    # Sort by bucket size (descending)
    bucket_info.sort(key=lambda x: x[1], reverse=True)

    # Print header
    print(f"{'Bucket #':<10} {'Images':<10} {'Target Size':<15} {'Aspect Ratio':<15}")
    print("-" * 50)

    # Print bucket information
    for i, (bucket_idx, num_images, target_size) in enumerate(bucket_info):
        if max_buckets is not None and i >= max_buckets:
            print(f"\n... and {len(bucket_info) - max_buckets} more buckets")
            break

        w, h = target_size
        aspect = w / h if h > 0 else 0
        print(f"{bucket_idx:<10} {num_images:<10} {w}x{h:<15} {aspect:.3f}")

    # Print histogram of bucket sizes
    # if bucket_info:
    #     bucket_sizes = [info[1] for info in bucket_info]
    #     size_counts = {}
    #     for size in bucket_sizes:
    #         if size in size_counts:
    #             size_counts[size] += 1
    #         else:
    #             size_counts[size] = 1
    #
    #     print("\nBucket Size Distribution:")
    #     print("------------------------")
    #     for size, count in sorted(size_counts.items()):
    #         print(f"Buckets with {size} images: {count}")


def save_model_safetensors(model, path):
    """Save model using safetensors format"""
    state_dict = model.state_dict()
    save_file(state_dict, path)
    print(f"Model saved to {path} using safetensors")


def load_model_safetensors(model, path):
    """Load model using safetensors format"""
    state_dict = load_file(path)
    model.load_state_dict(state_dict)
    print(f"Model loaded from {path} using safetensors")
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Image Aesthetic Prediction using VAE latents"
    )
    parser.add_argument("--image_dir", help="Directory containing images")
    parser.add_argument(
        "--csv_path",
        default=None,
        help="CSV file containing image metadata (images.metadata.csv)",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "predict"],
        required=True,
        help="Mode: train or predict",
    )
    parser.add_argument("--model_path", help="Path to save/load the aesthetic model")
    parser.add_argument(
        "--vae_model",
        default="black-forest-labs/FLUX.1-schnell",
        help="Hugging Face VAE model ID",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument("--ava_file", required=True, help="Path to AVA.txt file")
    parser.add_argument("--tag_file", help="Path to tags.txt file")
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Ratio of data to use for training",
    )
    parser.add_argument(
        "--learning_rate", default=0.001, help="Learning rate. (Default 0.001)"
    )
    args = parser.parse_args()

    # Load VAE model from Hugging Face
    vae_model = AutoencoderKL.from_pretrained(args.vae_model, subfolder="vae")

    # Set up data transformations
    # transform = transforms.Compose(
    #     [
    #         # transforms.Resize((256, 256)),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ]
    # )
    transform = v2.Compose(
        [
            # v2.RandomResizedCrop(size=(224, 224), antialias=True),
            v2.ToImage(),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Create dataset and dataloader
    print("Loading AVA dataset...")
    # dataloader, dataset = create_bucketed_ava_dataloader(
    #     image_dir=args.image_dir,
    #     ava_txt_path=args.ava_txt_path,
    #     batch_size=32,
    #     bucket_tolerance=32,  # Set to 8, 16, 32, or 64 as needed
    #     transform=transform,
    #     tag_file=args.tag_file,
    # )

    print(f"Batch size: {args.batch_size}")

    train_dataloader, test_dataloader, train_dataset, test_dataset = (
        create_bucketed_ava_dataloader_split(
            image_dir=args.image_dir,
            ava_txt_path=args.ava_file,
            batch_size=args.batch_size,
            bucket_tolerance=32,  # Set to 8, 16, 32, or 64 as needed
            transform=transform,
            tag_file=args.tag_file,
            csv_path=args.csv_path,
            train_ratio=args.train_ratio,
        )
    )

    # Print information about buckets in the training dataset
    print("Training Dataset Buckets:")
    print_bucket_info(train_dataset, max_buckets=20)  # Show top 20 buckets

    print("\nTest Dataset Buckets:")
    print_bucket_info(test_dataset, max_buckets=20)
    # full_dataset = AVADataset(args.image_dir, args.ava_file, transform, args.tag_file)

    # Split dataset
    # dataset_size = len(full_dataset)
    # train_size = int(args.train_ratio * dataset_size)
    # test_size = dataset_size - train_size
    #
    # train_dataset, test_dataset = torch.utils.data.random_split(
    #     full_dataset, [train_size, test_size]
    # )
    #
    # print(f"Dataset split: {train_size} training samples, {test_size} testing samples")
    #
    # # Create dataloaders
    # train_loader = DataLoader(
    #     train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    # )
    # test_loader = DataLoader(
    #     test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    # )

    accelerator = Accelerator()

    train_dataloader, test_dataloader, vae_model = accelerator.prepare(
        train_dataloader, test_dataloader, vae_model
    )

    # Get latent dimension - we need to process one batch to determine this
    sample_batch, _, _ = next(iter(train_dataloader))
    sample_batch = sample_batch
    sample_latents = extract_latents(vae_model, sample_batch)
    print(sample_latents.shape[2])
    # latent_dim = sample_latents.view(1, -1).size(1)
    latent_dim = sample_latents.shape[2]

    # Create or load aesthetic prediction model
    aesthetic_model = LatentAestheticsTransformer()

    aesthetic_model = accelerator.prepare(aesthetic_model)

    total_params = sum(p.numel() for p in aesthetic_model.parameters())
    print(f"Total trainable parameters: {total_params:,}")

    if args.mode == "train":
        print(
            f"Training aesthetic prediction model with {latent_dim}-dimensional latents"
        )
        aesthetic_model = train_aesthetic_model(
            train_dataloader,
            vae_model,
            aesthetic_model,
            num_epochs=args.epochs,
            lr=args.learning_rate,
        )

        # Save the trained model
        if args.model_path:
            model_path = args.model_path
            if not model_path.endswith(".safetensors"):
                model_path = f"{model_path}.safetensors"
            save_model_safetensors(aesthetic_model, model_path)

        # Evaluate on test set
        print("Evaluating on test set...")
        test_results = predict_aesthetic_scores(
            test_dataloader, vae_model, aesthetic_model
        )

        # Calculate performance metrics
        pred_scores = np.array([r[1] for r in test_results])
        true_scores = np.array([r[2] for r in test_results])

        mse = np.mean((pred_scores - true_scores) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(pred_scores - true_scores))

        print("Test set metrics:")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")

    elif args.mode == "predict":
        # Load pre-trained model
        if args.model_path:
            model_path = args.model_path
            if not model_path.endswith(".safetensors"):
                model_path = f"{model_path}.safetensors"
            aesthetic_model = load_model_safetensors(aesthetic_model, model_path)
        else:
            print("Warning: No model path provided. Using untrained model.")

        # Predict scores for all images
        results = predict_aesthetic_scores(test_dataloader, vae_model, aesthetic_model)

        # Print scores
        print("\nAesthetic Predictions:")
        for img_name, score in results:
            print(f"{img_name}: {score:.2f}/10")

        # Save results to file
        output_file = "aesthetic_scores.csv"
        with open(output_file, "w") as f:
            f.write("image,score\n")
            for img_name, score in results:
                f.write(f"{img_name},{score:.4f}\n")

        print(f"Results saved to {output_file}")

        # Print sample results
        print("\nSample predictions (Image ID, Predicted Score, True Score):")
        for img_id, pred, true in results[:10]:
            print(f"{img_id}: {pred:.2f}/10 (True: {true:.2f}/10)")


if __name__ == "__main__":
    main()
