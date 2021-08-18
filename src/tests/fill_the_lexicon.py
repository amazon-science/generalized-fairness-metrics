from typing import Set, Callable, List, Tuple, Dict
from expanded_checklist.checklist.editor import Editor
import regex as re

import src.config as cfg
from src.utils import get_name_from_path, get_file_paths
from lemminflect import getInflection

import os
import logging
from munch import Munch
import csv
from itertools import product, chain, combinations
import pandas as pd
from collections import defaultdict

logging.basicConfig(level=cfg.log_level)
logger = logging.getLogger(__name__)


def has_non_latin_chars(x: str) -> bool:
    return re.search(r'[^\p{Latin}]', x) is None


def enhance_lexicon(
    editor: Editor,
    lkey: str,
    fpath: str
) -> None:
    """
    A function that can be used to expand the lists of terms for a key
    that is already present in the lexicon.
    """
    logger.info(f"Enhancing {lkey}...")
    logger.info(f"Initial size: {len(editor.lexicons[lkey])}")

    new_terms: Set[str] = read_terms(fpath)
    new_terms.update(editor.lexicons[lkey])

    editor.add_lexicon(lkey, list(new_terms), overwrite=True)
    logger.info(f"Final size: {len(editor.lexicons[lkey])}")


#########################
#   READING TERMS       #
#########################

def read_terms_from_file(
    fpath: str,
    fun: Callable[[str], str] = None
) -> Set[str]:
    with open(fpath, "r") as f:
        lines = f.readlines()

        def map_fun(x):
            if fun:
                x = fun(x)
            return x.strip().strip('\u200e')
        return set(map(map_fun, lines))


def read_terms(
    name: str,
    fun: Callable[[str], str] = None,
    root_dir: str = cfg.identity_terms_dir_path
) -> Set[str]:
    if os.path.isfile(name):
        return read_terms_from_file(name, fun)

    name = name.rstrip(".txt")
    terms_fpaths = get_file_paths(root_dir)
    for fp in terms_fpaths:
        fname = get_name_from_path(fp)
        if fname == name:
            return read_terms_from_file(fp, fun)

    logger.error(f"Couldn't find a terms file with the name: {name}.")
    return set()


def read_csv_terms(
    editor: Editor,
    path: str,
    overwrite: bool = False
) -> None:
    terms_name = get_name_from_path(path)

    # the underscores are reserved for splitting queries: e.g. gender_adj
    # means we query for adjectives in the gender data frame
    terms_name = terms_name.replace("_", "-")
    df = pd.read_csv(path, index_col="TERM")

    for p in df.columns:
        if all([(re.match(r'^[0-9]+$', str(x))
                is not None) or x == "" for x in df[p].unique()]):
            df[p] = df[p].apply(int)

    # don't allow spaces for property values (but allow for terms)
    df = df.apply(
        lambda x: [y.replace(" ", "-") if y and type(y) == str
                   else y for y in x], axis="index")

    if "census" in path:
        df.index = df.index.map(lambda x: x[0].upper() + x[1:].lower())

    editor.add_lexicon(terms_name, df, overwrite=overwrite)


def read_all_terms_into_lexicon(
    editor: Editor,
    fun: Callable[[str], str] = None,
    overwrite: bool = False,
    root_dir: str = cfg.identity_terms_dir_path
) -> None:
    """
    The function looks for all files ending with .txt and .csv in the given
    directory and reads them into the lexicon under they key that is the same
    as the file's name.
    """
    terms_fpaths = get_file_paths(root_dir)
    skip = ['matched_', 'pronouns', 'SOURCES']

    for fp in terms_fpaths:
        if not all([x not in fp for x in skip]):
            continue

        if ".txt" in fp:
            terms: List[str] = list(read_terms_from_file(fp, fun))
            try:
                editor.add_lexicon(
                    get_name_from_path(fp), terms, overwrite=overwrite)
            except Exception:
                logger.error(f"Exception on {fp}")
        elif ".csv" in fp:
            try:
                read_csv_terms(editor, fp, overwrite=overwrite)
            except Exception:
                logger.error(f"Exception on {fp}")


def read_terms_into_lexicon(
    editor: Editor,
    name: str,
    name_in_lex: str = "",  # how to save these terms in the lexicon
    overwrite: bool = False,
    fun: Callable[[str], str] = None
) -> bool:
    """
    This function can be used to read .txt list of terms into the lexicon
    with a specific name
    """
    if not name_in_lex:
        name_in_lex = name

    terms: Set = read_terms(name, fun)
    if not terms:
        return False
    else:
        try:
            editor.add_lexicon(name_in_lex, list(terms), overwrite=overwrite)
        except Exception:
            return False
        return True


##################################
#       FILLING THE LEXICON     #
##################################


def fill_the_lexicon(editor: Editor) -> None:
# lexicons provided by checklist:
#  dict_keys(['male', 'female', 'first_name', 'first_pronoun', 'last_name', 'country',
# 'nationality', 'city', 'religion', 'religion_adj', 'sexual_adj', 'country_city',
# 'male_from', 'female_from', 'last_from'])

    # add female names from wiktionary
    fem_names: Set[str] = read_terms("only_female_names")
    editor.add_lexicon(
        "english_female_given_names", list(fem_names), overwrite=True)
    for _, names in editor.lexicons['female_from'].items():
        fem_names.update(names)

    # filter the names with non-latin characters
    fem_names = [name for name in fem_names if has_non_latin_chars(name)]
    editor.add_lexicon("all_female_names", fem_names, overwrite=True)

    # add male names from wiktionary
    male_names: Set[str] = read_terms("only_male_names")
    editor.add_lexicon(
        "english_male_given_names", list(male_names), overwrite=True)
    for _, names in editor.lexicons['male_from'].items():
        male_names.update(names)
    male_names = [name for name in male_names if has_non_latin_chars(name)]
    editor.add_lexicon("all_male_names", male_names, overwrite=True)

    # read the remaining lists into the lexicon, with their default names
    read_all_terms_into_lexicon(
        editor, root_dir=cfg.identity_terms_dir_path, overwrite=True)
