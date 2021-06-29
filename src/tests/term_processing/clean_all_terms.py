import os
import sys
import argparse
import logging
from typing import Set
import pathlib

curdir = pathlib.Path(__file__).parent.absolute()
sys.path.insert(0, f"{curdir}/../../../")

from src.utils import read_terms, get_file_paths
from src.tests.term_processing.filter_terms import filter_terms
import src.config as cfg

logging.basicConfig(level=cfg.log_level)
logger = logging.getLogger(__name__)


def main(args) -> None:
    if not os.path.isdir(args.dir_to_clean):
        logger.error(f"Directory {args.dir_to_clean} does not exist.")
        sys.exit()

    if not os.path.isdir(args.dir_with_filter_terms):
        logger.error(f"Directory {args.dir_with_filter_terms} does not exist.")
        sys.exit()

    fpaths_to_clean = get_file_paths(
        args.dir_to_clean,
        not_in_dir_name=[args.dir_with_filter_terms.rstrip("/")])
    filter_fpaths = get_file_paths(args.dir_with_filter_terms)

    for fpath in fpaths_to_clean:
        logger.info(f"Filtering path {fpath}...")
        filter_terms(fpath, filter_fpaths, fpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir-to-clean', type=str)
    parser.add_argument('--dir-with-filter-terms', type=str)
    args = parser.parse_args()

    main(args)
