import csv
import os
from shutil import copyfile
import argparse


def main(args):
    input = args.input
    output = args.output
    filter = args.score

    assert os.path.exists(input)

    os.makedirs(output, exist_ok=True)

    results = []

    with open(input, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)

    filtered_results = []

    for result in results:
        if float(result["score"]) > filter:
            filtered_results.append(result)

    for result in filtered_results:
        copyfile(result["file"], os.path.join(output, os.path.basename(result["file"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input", help="Input CSV of format file, score with full file path"
    )

    parser.add_argument("--score", type=float, help="Score to filter by")
    parser.add_argument(
        "--output", type=str, help="Where do we place the selected files?"
    )

    args = parser.parse_args()

    main(args)

# open scores.csv
# filter by filter
# take resulting list and copy files into output
