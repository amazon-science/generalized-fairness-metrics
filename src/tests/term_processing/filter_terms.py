import os
import sys
import argparse
import logging
from typing import Set
import pathlib
from typing import List

curdir = pathlib.Path(__file__).parent.absolute()
sys.path.insert(0, f"{curdir}/../../../")

from src.utils import read_terms

logger = logging.getLogger(__name__)


def filter_terms(
    terms_path: str,
    filter_paths: List[str],
    out_path: str
) -> None:
    if not os.path.isfile(terms_path):
        logger.error(f"File {terms_path} does not exist.")
        sys.exit()

    if not os.path.exists(os.path.dirname(out_path)):
        dirname = os.path.dirname(out_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

    orig_terms: Set[str] = read_terms(terms_path)

    terms_to_remove: Set[str] = set()
    for fp in filter_paths:
        terms_to_remove.update(read_terms(fp))

    terms_to_keep = orig_terms.difference(terms_to_remove)
    with open(out_path, "w") as f:
        for term in terms_to_keep:
            f.write(f"{term}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--terms-path', type=str)
    parser.add_argument('--filter-paths', nargs='*', type=str,
                        dest='filter_paths')
    parser.add_argument('--out-path', type=str)
    args = parser.parse_args()
    filter_terms(args.terms_path, args.filter_paths, args.out_path)
