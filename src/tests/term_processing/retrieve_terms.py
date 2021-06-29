import logging
import argparse
import os
from typing import Set, Optional
import pathlib

os.environ["PYWIKIBOT_NO_USER_CONFIG"] = "1"
import pywikibot  # noqa

parser = argparse.ArgumentParser()
parser.add_argument(
    '--category', type=str, default='Category:English female given names')
parser.add_argument('--out-path', type=str, default=None)
args = parser.parse_args()

site = pywikibot.Site('en', 'wiktionary')

logger = logging.getLogger(__name__)


def retrieve_words(
    category: str,
    limit: Optional[int] = None
) -> Set[str]:

    cat = pywikibot.Category(site, category)
    result: Set[str] = set()
    for i in cat.members():
        title: str = i.title()
        if "Category:" in title:
            continue

        result.add(title)

        if limit and len(result) == limit:
            break
    return result


def get_out_path(category: str) -> str:
    full_name = category.rsplit(":", 1)[-1]
    full_name = full_name.lower()
    full_name = full_name.replace(" ", "_")
    curdir = pathlib.Path(__file__).parent.absolute()
    return f"{curdir}/../../term_lists/{full_name}.txt"


def main() -> None:
    terms: Set[str] = retrieve_words(args.category)

    if args.out_path is None:
        args.out_path = get_out_path(args.category)

    if not os.path.exists(os.path.dirname(args.out_path)):
        dirname = os.path.dirname(args.out_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

    with open(args.out_path, "w") as f:
        for term in terms:
            f.write(f"{term}\n")


if __name__ == "__main__":
    main()
