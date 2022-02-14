
import csv
from typing import List, Set
from lemminflect import getInflection
import regex as re

from expanded_checklist.checklist.editor import Editor
from collections import OrderedDict
import pandas as pd

import logging
import src.config as cfg

logging.basicConfig(level=cfg.log_level)
logger = logging.getLogger(__name__)


def get_templates(
    editor: Editor,
    path: str,
    # by default return templates with all identity terms
    identity_keys: set = {'identity_np', 'identity_adj', 'person'}
) -> pd.DataFrame:
    """
    If identity keys set is empty then assume all items not in lexicon
    """
    template_rows = []
    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        polarity_phrases = []

        for row in reader:
            # handle maybes, marked with '?' (as in regex)
            # atm there is support for only one question marked statement
            maybe = re.search(r'{[^}]*\?}', row['TEMPLATE'])

            # get different inflections of a verb
            # all features to return have to be listed ahead of the verb, e.g.
            # {To_pst_prs_sg_pl_be}, if not it defaults to sg, pl, prs
            # see get_paradigms for supported tenses
            verb_infl = re.search(
                r'{((?:[Nn]ot)?[tT]o)_([^:}]+):([^:}]+)}', row['TEMPLATE'])
            polarity_appendix = re.search(r'{neut}', row['TEMPLATE'])

            if maybe:
                resolve_maybe(template_rows, row)
            elif verb_infl:
                resolve_infl(template_rows, row, verb_infl)
            elif polarity_appendix:
                polarity_phrases.append(row)
            else:
                template_rows.append(row)

    # make sure all keys are in the lexicon or in the identity keys provided
    # filter those sentences that have keys that are not in either
    # (and alert the user that has been the case)
    template_rows = process_all_keys(template_rows, editor, identity_keys)
    if not template_rows:
        return None
    df = pd.DataFrame(template_rows, columns=template_rows[0].keys())
    return df


def process_all_keys(
    template_rows: List[OrderedDict],
    editor: Editor,
    identity_keys: Set[str]
) -> List[OrderedDict]:
    """
    Return only sentences that have in them one of the provided identity
    keys and for each all other keys are in the lexicon
    """
    new_template_rows = []
    missing_keys = set()

    for row in template_rows:
        sent = row["TEMPLATE"]
        keys = re.findall(r"{[^}]+}", sent)

        sent_missing_keys = set()

        cleaned_keys = []
        for key in keys:
            low_key = key[1:-1][0].lower() + key[1:-1][1:]
            low_key = re.sub(r"^[a-z]+:", "", low_key)
            cleaned_keys.append(low_key)

        # filter out sentences without the identity
        if all(ident not in cleaned_keys for ident in identity_keys):
            continue

        for key, clean_key in zip(keys, cleaned_keys):
            if clean_key in editor.lexicons or clean_key in identity_keys:
                continue
            try:
                editor.template(key)
            except Exception:
                sent_missing_keys.add(key)
        if sent_missing_keys:
            missing_keys.update(sent_missing_keys)
        else:
            new_template_rows.append(row)

    # missed = len(template_rows) - len(new_template_rows)
    # logger.info(f"Ommited {missed} sentences out of {len(template_rows)}.")
    if missing_keys:
        logger.warning(f"Keys not in lexicon/not identity {missing_keys}.")
    return new_template_rows


def resolve_maybe(
    template_rows: List[OrderedDict],
    row: OrderedDict
) -> None:
    new_row = row.copy()
    new_row["TEMPLATE"] =\
        re.sub(r' ?{[^}]*\?}', '', new_row["TEMPLATE"])
    template_rows.append(new_row)

    row["TEMPLATE"] = re.sub(r'\?}', '}', row["TEMPLATE"])
    template_rows.append(row)


def resolve_infl(
    template_rows: List[OrderedDict],
    row: OrderedDict,
    regex_match
) -> None:
    verb_infl = regex_match

    to = verb_infl.group(1)
    verb = verb_infl.group(2)
    morph = verb_infl.group(3)

    msplits = morph.split("_")
    neg = to.lower() == "notto"

    num = ["sg", "pl"]
    if "sg" in msplits:
        num = ["sg"]
    elif "pl" in msplits:
        num = ["pl"]

    potential_tenses =\
        ["prs", "fut", "pst", "prscont", "prsperf", "pstcont"]
    tenses = []
    for pt in potential_tenses:
        if pt in msplits:
            tenses.append(pt)
    if not tenses:
        tenses = ["prs"]

    par = get_paradigm(verb, num, tenses, neg)
    for x in par:
        if to[0].isupper():
            x = x.capitalize()
        new_row = row.copy()
        new_row["TEMPLATE"] = re.sub(
            r'{((?:[Nn]ot)?[tT]o)_([^:}]+):([^:}]+)}',
            x, new_row["TEMPLATE"])
        template_rows.append(new_row)


#########################################

def get_df_with_sent_class(df: pd.DataFrame, class_: str) -> pd.DataFrame:
    return filter_df(df, "SENT", class_)


def filter_df(df: pd.DataFrame, feat: str, val: str) -> pd.DataFrame:
    return df[df[feat].str.match(val)]


def df2list(df: pd.DataFrame):
    return list(df['TEMPLATE'])


#########################
#       HELPERS         #
#########################

def get_paradigm(
    verb: str,
    numbers: List[str] = ["sg", "pl"],
    # ["prs", "fut", "pst", "prscont", "prsperf", "pstcont"]
    tenses: List[str] = ["prs"],
    neg: bool = False
) -> List[str]:

    vbd = getInflection(verb, tag='VBD')[0]
    vbg = getInflection(verb, tag='VBG')[0]
    vbn = getInflection(verb, tag='VBN')[0]
    vbp = getInflection(verb, tag='VBP')[0]
    vbz = getInflection(verb, tag='VBZ')[0]

    sg, pl = set(), set()
    non_p_3_sg = ["I", "you"]
    p_pl = ["we", "they", "you"]
    p_3_sg = ["he", "she"]

    to_be_prs_sg = ["I am", "you are", "he is", "she is"] if not neg else \
        ["I'm not", "you aren't", "he isn't", "she isn't"]
    to_be_prs_pl = ["we are", "you are", "they are"] if not neg else \
        ["we aren't", "you aren't", "they aren't"]

    to_be_prs_sg_pst =\
        ["I was", "you were", "he was", "she was"] if not neg else \
        ["I wasn't", "you weren't", "he wasn't", "she wasn't"]
    to_be_prs_pl_pst = ["they were", "you were", "we were"] if not neg else \
        ["they weren't", "you weren't", "we weren't"]

    if "prs" in tenses:
        if verb == "be":
            sg.update(to_be_prs_sg)
            pl.update(to_be_prs_pl)
        else:
            k1 = " don't " if neg else " "
            k2 = " doesn't " if neg else " "
            vbz1 = vbp if neg else vbz
            sg.update(
                [f"{x}{k1}{vbp}" for x in non_p_3_sg] +
                [f"{x}{k2}{vbz1}" for x in p_3_sg])
            pl.update([f"{x}{k1}{vbp}" for x in p_pl])
    if "pst" in tenses:
        if verb == "be":
            sg.update(to_be_prs_sg_pst)
            pl.update(to_be_prs_pl_pst)
        else:
            k = " didn't " if neg else " "
            vbd1 = vbp if neg else vbd
            sg.update([f"{x}{k}{vbd1}" for x in non_p_3_sg + p_3_sg])
            pl.update([f"{x}{k}{vbd1}" for x in p_pl])
    if "fut" in tenses:
        k = "won't" if neg else "will"
        sg.update([f"{x} {k} {verb}" for x in non_p_3_sg + p_3_sg])
        pl.update([f"{x} {k} {verb}" for x in p_pl])
    if "prscont" in tenses:
        sg.update([f"{x} {vbg}" for x in to_be_prs_sg])
        pl.update([f"{x} {vbg}" for x in to_be_prs_pl])
    if "pstcont" in tenses:
        sg.update([f"{x} {vbg}" for x in to_be_prs_sg_pst])
        pl.update([f"{x} {vbg}" for x in to_be_prs_pl_pst])
    if "prsperf" in tenses:
        k1 = "haven't " if neg else "have"
        k2 = "hasn't " if neg else "has"
        sg.update([f"{x} {k1} {vbn}" for x in non_p_3_sg] +
                  [f"{x} {k2} {vbn}" for x in p_3_sg])
        pl.update([f"{x} {k1} {vbn}" for x in p_pl])

    to_ret = []
    if "sg" in numbers:
        to_ret += list(sg)
    if "pl" in numbers:
        to_ret += list(pl)
    return to_ret
