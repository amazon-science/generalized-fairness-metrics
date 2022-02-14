import os
import regex as re
from typing import List

import src.config as cfg
import logging

logging.basicConfig(level=cfg.log_level)
logger = logging.getLogger(__name__)


def get_name_from_path(fpath: str) -> str:
    """
    Removes all directories from the path and returns the name of the
    file (without the extension).
    """
    return fpath.split("/")[-1].rsplit(".", 1)[0]


def join_tokens(
    toks: List[str],
    clean: bool = True
) -> str:
    try:
        text = ' '.join([x.text for x in toks])
    except AttributeError:
        text = ' '.join(toks)

    if clean:
        text = re.sub(r"\s([:,.!?'\)>\]}\-\"])", r"\g<1>", text)
        text = re.sub(r"([\-\(\(<\[{`])\s", r"\g<1>", text)
        text = re.sub(r" (n't|'s|'t|\.)", r'\g<1>', text)
    return text


#####################
#   GETTING PATHS   #
#####################

def get_file_paths(
    root_path: str,
    in_name: List[str] = [],
    in_dir_name: List[str] = [],
    not_in_dir_name: List[str] = [],
    not_in_file_name: List[str] = []
) -> List[str]:
    fnames: List[str] = []
    for root, _, fs in os.walk(root_path.rstrip("/") + "/", topdown=True):
        if not all([indir in root for indir in in_dir_name]) or \
                    any([n in root for n in not_in_dir_name]):
            continue

        for name in fs:
            if all([x in name for x in in_name]) and \
                    not any([n in name for n in not_in_file_name]):
                fnames.append(root.rstrip("/") + "/" + name)
    return fnames


def get_dir_paths(
    root_path: str,
    in_final_dir_name: List[str] = []
) -> List[str]:
    fnames: List[str] = []
    for root, dirs, _ in os.walk(root_path.rstrip("/") + "/", topdown=True):
        for name in dirs:
            if all([indir in name for indir in in_final_dir_name]):
                fnames.append(root.rstrip("/") + "/" + name)
    return sorted(fnames)
