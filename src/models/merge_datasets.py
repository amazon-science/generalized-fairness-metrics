import argparse
import os
import sys

sys.path.insert(0, f"../")
sys.path.insert(0, f".")


from src.models.readers.read_data import read_data
import src.config as cfg

from typing import List
import random

import logging
logger = logging.getLogger(__name__)


def get_data_for_split(split: str, len_threshold: int) -> List[str]:
    all_lines = []

    for dataset in ["sst-2", "semeval-2", "imdb", "rotten_tomatoes",
                    "yelp_polarity"]:
        sents, labels, _ = read_data(dataset, split)

        zipped = list(zip(sents, labels))

        # filter examples longer than len_threshold words
        zipped = [(sent, label) for (sent, label) in zipped
                  if len(sent.split(" ")) < len_threshold]

        # some datasets are much larger than others -- hence the filtering
        # to even out representation of different domains
        if len(zipped) > 1500:
            zipped = random.sample(zipped, 1500)

        for text, label in zipped:
            # format supported by simple reader
            line = f"{dataset}\t{label}\t{text}"
            all_lines.append(line)
    random.shuffle(all_lines)
    return all_lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', type=str)
    parser.add_argument('--len-threshold', type=int, default=250)
    args = parser.parse_args()

    odir = args.out_dir.rstrip("/")
    if not os.path.exists(odir):
        os.makedirs(odir, exist_ok=True)

    for split in ["test", "dev", "train"]:
        all_lines = get_data_for_split(split, args.len_threshold)
        data_name = "mixed_sst_semeval_imdb_rt_yelp"
        with open(f"{odir}/{data_name}_{split}.txt", "w") as f:
            for line in all_lines:
                f.write(f"{line}\n")


if __name__ == "__main__":
    main()
