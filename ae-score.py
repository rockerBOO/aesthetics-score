import os.path

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import numpy as np
from PIL import Image
import math
import os
import sys
import traceback
from platformdirs import user_cache_dir
import argparse
import csv
import time

APP_NAME = "ae-score"
APP_AUTHOR = "rockerBOO"

model_to_host = {
    "chadscorer": "https://github.com/grexzen/SD-Chad/raw/main/chadscorer.pth",
    "sac+logos+ava1-l14-linearMSE": "https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/main/sac+logos+ava1-l14-linearMSE.pth?raw=true",
}

MODEL = "chadscorer"


def ensure_model(model):
    """
    Make sure we have the model to score with.
    Saves into your cache directory on your system
    """
    cache_dir = user_cache_dir(APP_NAME, APP_AUTHOR)

    Path(cache_dir).mkdir(exist_ok=True)

    file = MODEL + ".pth"
    full_file = os.path.join(cache_dir, file)
    if not Path(full_file).exists():
        url = model_to_host[MODEL]
        import requests

        print(f"downloading {url}")
        r = requests.get(url)
        print(r)
        with open(full_file, "wb") as f:
            f.write(r.content)
            print(f"saved to {full_file}")


def clear_cache():
    """
    Removes all the cached models
    """
    cache_dir = user_cache_dir(APP_NAME, APP_AUTHOR)

    for f in os.listdir(cache_dir):
        os.remove(os.path.join(cache_dir, f))


def load_model(model, device="cpu"):
    ensure_model(model)
    cache_dir = user_cache_dir(APP_NAME, APP_AUTHOR)

    pt_state = torch.load(
        os.path.join(cache_dir, model + ".pth"), map_location=torch.device("cpu")
    )

    # CLIP embedding dim is 768 for CLIP ViT L 14
    predictor = AestheticPredictor(768)
    predictor.load_state_dict(pt_state)
    predictor.to(device)
    predictor.eval()

    return predictor


def load_clip_model(device="cpu"):
    clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)

    return clip_model, clip_preprocess


class AestheticPredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # load the model you trained previously or the model available in this repo

    predictor = load_model(MODEL, device)
    clip_model, clip_preprocess = load_clip_model(device=device)

    def get_image_features(
        image: Image.Image, device="cpu", model=clip_model, preprocess=clip_preprocess
    ):
        image = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
            # l2 normalize
            image_features /= image_features.norm(dim=-1, keepdim=True)
        image_features = image_features.cpu().detach().numpy()
        return image_features

    def get_score(predictor, image, device="cpu"):
        image_features = get_image_features(image, device)
        score = predictor(torch.from_numpy(image_features).to(device).float())
        return score.item()

    scores = []

    input_images = Path(args.image_file_or_dir)
    if input_images.is_dir():
        for file in os.listdir(input_images):
            image = Image.open(os.path.join(input_images, file))
            scores.append({"file": file, "score": get_score(predictor, image, device)})
    else:
        image = Image.open(args.image_file_or_dir)
        chad_score = get_score(predictor, image, device)
        scores.append(
            {
                "file": args.image_file_or_dir,
                "score": get_score(predictor, image, device),
            }
        )

    sorted_scores = sorted(scores, key=lambda x: x["score"], reverse=True)

    print(args)
    if args.save_csv:
        fieldnames = ["file", "score"]
        id = str(round(time.time()))
        with open(f"scores-{id}.csv", "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for score in sorted_scores:
                writer.writerow(score)

    for score in sorted_scores:
        print(score["file"], score["score"])

    acc_scores = 0
    for score in sorted_scores:
        acc_scores = acc_scores + score["score"]

    print(f"average score: {acc_scores / len(sorted_scores)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "image_file_or_dir",
        type=str,
        help="Image file or directory containing the images",
    )

    parser.add_argument(
        "--save_csv",
        default=False,
        action="store_false",
        help="Save the results to a csv file in the current directory",
    )

    args = parser.parse_args()

    main(args)
