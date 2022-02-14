import sys
sys.path.insert(0, ".")

import src.tests.process_templates as tp
from src.tests.test_creation_utils import get_grouped_data_from_template
from src.tests.save_and_load import save_suite
from src.tests.process_templates import get_templates
from expanded_checklist.checklist.editor import Editor
from expanded_checklist.checklist.test_suite import TestSuite

from typing import Dict, List
from expanded_checklist.checklist.eval_core import EvaluationCore
from src.tests.fill_the_lexicon import fill_the_lexicon
from munch import Munch

import src.config as cfg

from collections import defaultdict
import pandas as pd

import logging

logging.basicConfig(level=cfg.log_level)
logger = logging.getLogger(__name__)


#####################
#   KEY FUNCITON   #
#####################

def create_grouped_explicit_identity_suite(
    editor: Editor,
    name: str,
    tpath2slot2group2terms: str,
    sensitive_attr: str,
    nsamples: int = None
):
    """
    Use both adjectives and nps.
    """
    suite = TestSuite()

    full_data, full_meta, full_lab_dict = [], [], defaultdict(Munch)

    group_keys = None
    for tpath, slot2group2terms in tpath2slot2group2terms.items():
        for slot, group2terms in slot2group2terms.items():
            df_templates = get_templates(
                editor, tpath, identity_keys={slot})
            if df_templates is None:
                continue

            if group_keys is None:
                group_keys = group2terms.keys()
                logger.info(
                    f"\n{name} suite, group keys:\n{sorted(group_keys)}")
            # the evaluation code assumes exactly the same groups
            # per each example
            assert group_keys == group2terms.keys()

            data, meta, lab_dict =\
                get_grouped_data_from_template(
                    editor=editor,
                    df_templates=df_templates,
                    nsamples=nsamples,
                    identity_key=slot,
                    groups=group2terms
                )
            full_data += data
            full_meta += meta

            update_lab_dict(full_lab_dict, lab_dict)

    full_lab_dict = Munch(full_lab_dict)
    ct = EvaluationCore(
        full_data,
        full_lab_dict,
        meta=full_meta,
        name=f"{name}", capability=sensitive_attr)
    suite.add(ct)

    save_suite(suite, name)
    logger.info(f"\nCreated suite: {name}, #sents: {len(full_data)}")
    return suite, name, full_data


####### helpers

def update_lab_dict(full_lab_dict, new_lab_dict):
    for task in cfg.supported_tasks:
        if task in new_lab_dict:
            if 'labels' not in full_lab_dict[task]:
                full_lab_dict[task]['labels'] = []
            full_lab_dict[task]['labels'] += new_lab_dict[task]['labels']
            # classes should be the same for all identity keys
            # -- the same template set is used
            full_lab_dict[task]['n_classes'] = new_lab_dict[task]['n_classes']


def fill_from_adjs(editor, adj_term_list, group, np_groups, adj_groups):
    for term in adj_term_list:
        if not term or type(term) != str:
            continue
        np_groups[group].append(f'{term} person')
        adj_groups[group].append(term)


def fill_from_pps(editor, pp_term_list, group, np_groups, pp_groups):
    for term in pp_term_list:
        if not term or type(term) != str:
            continue
        np_groups[group].append(f'person {term}')
        pp_groups[group].append(term)


def get_person_group(editor, person_str, pp_groups=None, adj_groups=None):
    if pp_groups is None and adj_groups is None:
        return None

    to_ret = defaultdict(list)
    if pp_groups is not None:
        for group, terms in pp_groups.items():
            for term in terms:
                to_ret[group].append(f'{person_str} {term}')

    if adj_groups is not None:
        for group, terms in adj_groups.items():
            for term in terms:
                to_ret[group].append(f'{term} {person_str}')
    return to_ret


def merge_groups(editor, group1, group2):
    to_ret = defaultdict(list)
    for group, terms in group1.items():
        to_ret[group] += terms
    for group, terms in group2.items():
        to_ret[group] += terms
    return to_ret


#####################################
#   SUITES BY PROTECTED ATTRIBUTE   #
#####################################

def get_nationality_suite(
        editor, group_key="REGION", nsamples=None, name=None):
    # step 1: get the terms
    adj_key = "COUNTRY_ADJ"

    df = editor.lexicons['countries']
    if group_key in ["GDP_PPP", "GDP_NOMINAL"]:
        new_key = f'{group_key}_quantile'
        df[new_key] =\
            pd.qcut(df[group_key], q=6, labels=[f"{k}q" for k in range(1, 7)])
        group_key = new_key

    np_groups = defaultdict(list)
    adj_groups = defaultdict(list)
    country_groups = defaultdict(list)

    for group in df[group_key].unique():
        if not group or type(group) != str:
            continue
        agroup = df[df[group_key] == group]
        group = group.lower()
        fill_from_adjs(
            editor, list(agroup[adj_key]), group, np_groups, adj_groups)
        country_groups[group] = list(agroup.index)

    tpath2slot2group2terms = {
        # slots in the nationality templates
        cfg.nationality_template_path: {
            "person": np_groups,
            "country_adj": adj_groups,
            "country": country_groups
        },
        # slots in the generic templates
        cfg.generic_template_path: {
            "identity_adj": adj_groups,
            "identity_np": np_groups
        }
    }

    if not name:
        name = 'grouped_country_by_' + group_key.lower()
    s, n, data = create_grouped_explicit_identity_suite(
        editor, name, tpath2slot2group2terms, name, nsamples=nsamples)
    return s, n, data


def get_religion_suite(editor, group_key="GROUP", nsamples=None, name=None):
    df = editor.lexicons['religion']

    np_groups = defaultdict(list)
    adj_groups = defaultdict(list)
    religion_groups = defaultdict(list)

    for group in df[group_key].unique():
        if not group or type(group) != str:
            continue

        agroup = df[df[group_key] == group]
        group = group.lower()

        adjs = agroup[agroup['POS'] == 'adj']
        fill_from_adjs(editor, list(adjs.index), group, np_groups, adj_groups)
        nouns = agroup[agroup['POS'] == 'n']
        person_nouns = nouns[nouns['SEM'] == 'person']
        np_groups[group] += [term for term in person_nouns.index
                             if term and type(term) == str]
        religion_nouns = nouns[nouns['SEM'] == 'religion']
        religion_groups[group] += [term for term in religion_nouns.index
                                   if term and type(term) == str]

    tpath2slot2group2terms = {
        cfg.religion_template_path: {
            "person": np_groups,
            "religion_adj": adj_groups,
            "religion": religion_groups
        },
        cfg.generic_template_path: {
            "identity_adj": adj_groups,
            "identity_np": np_groups
        }
    }

    if not name:
        name = 'grouped_religion'
    s, n, data = create_grouped_explicit_identity_suite(
        editor, name, tpath2slot2group2terms, name, nsamples=nsamples)
    return s, n, data


def get_race_suite(editor, group_key="GROUP", nsamples=None, name=None):
    df = editor.lexicons['race']

    adj_groups = defaultdict(list)
    np_groups = defaultdict(list)

    for group in df[group_key].unique():
        if not group or type(group) != str:
            continue

        agroup = df[df[group_key] == group]
        group = group.lower()
        fill_from_adjs(
            editor, list(agroup.index), group, np_groups, adj_groups)

    tpath2slot2group2terms = {
        cfg.generic_template_path: {
            "identity_adj": adj_groups,
            "identity_np": np_groups
        },
        cfg.ethnicity_template_path: {
            "identity_adj": adj_groups,
            "identity_np": np_groups
        }
    }

    if not name:
        name = 'grouped_race'
    s, n, data = create_grouped_explicit_identity_suite(
        editor, name, tpath2slot2group2terms, name, nsamples=nsamples)
    return s, n, data


def get_age_suite(editor, group_key="GROUP", nsamples=None, name=None):
    df = editor.lexicons['age']
    pos_key = "POS"

    adj_groups = defaultdict(list)
    np_groups = defaultdict(list)

    for group in df[group_key].unique():
        if not group or type(group) != str:
            continue

        agroup = df[df[group_key] == group]
        group = group.lower()
        adjs = agroup[agroup[pos_key] == "adj"]
        fill_from_adjs(editor, list(adjs.index), group, np_groups, adj_groups)

    tpath2slot2group2terms = {
        cfg.generic_template_path: {
            "identity_adj": adj_groups,
            "identity_np": np_groups
        },
        cfg.age_template_path: {
            "identity_adj": adj_groups,
            "identity_np": np_groups
        }
    }

    if not name:
        name = 'grouped_age'
    s, n, data = create_grouped_explicit_identity_suite(
        editor, name, tpath2slot2group2terms, name, nsamples=nsamples)
    return s, n, data


def get_sexuality_suite(editor, group_key="GROUP", nsamples=None, name=None):
    df = editor.lexicons['sexuality']
    pos_key = "POS"

    np_groups = defaultdict(list)
    adj_groups = defaultdict(list)

    for group in df[group_key].unique():
        if not group or type(group) != str:
            continue

        agroup = df[df[group_key] == group]
        group = group.lower()

        adjs = agroup[agroup[pos_key] == "adj"]
        fill_from_adjs(editor, list(adjs.index), group, np_groups, adj_groups)
        nouns = agroup[agroup[pos_key] == "n"]
        np_groups[group] += [term for term in nouns.index
                             if term and type(term) == str]

    tpath2slot2group2terms = {
        cfg.generic_template_path: {
            "identity_adj": adj_groups,
            "identity_np": np_groups},
        cfg.gend_sex_template_path: {
            "identity_adj": adj_groups,
            "identity_np": np_groups},
        }

    if not name:
        name = 'grouped_sexuality'
    s, n, data = create_grouped_explicit_identity_suite(
        editor, name, tpath2slot2group2terms, name, nsamples=nsamples)
    return s, n, data


def get_gender_suite(editor, group_key="GROUP", nsamples=None, name=None):
    df = editor.lexicons['gender']
    pos_key = "POS"

    np_groups = defaultdict(list)
    adj_groups = defaultdict(list)

    for group in df[group_key].unique():
        if not group or type(group) != str:
            continue

        agroup = df[df[group_key] == group]
        group = group.lower()

        adjs = agroup[agroup[pos_key] == "adj"]
        fill_from_adjs(editor, list(adjs.index), group, np_groups, adj_groups)
        nouns = agroup[agroup[pos_key] == "n"]
        np_groups[group] += [term for term in nouns.index
                             if term and type(term) == str]

    tpath2slot2group2terms = {
        cfg.generic_template_path: {
            "identity_adj": adj_groups,
            "identity_np": np_groups},
        cfg.gend_sex_template_path: {
            "identity_adj": adj_groups,
            "identity_np": np_groups
        },
    }

    if not name:
        name = 'grouped_gender'
    s, n, data = create_grouped_explicit_identity_suite(
        editor, name, tpath2slot2group2terms, name, nsamples=nsamples)
    return s, n, data


def get_names_suite(editor, nsamples=100, name=None):
    name_groups = {
        "male": editor.lexicons['male'],
        "female": editor.lexicons['female']
    }

    tpath2slot2group2terms = {
        cfg.people_template_path: {
            "person": name_groups
        }
    }

    if not name:
        name = 'names_gender'
    s, n, data = create_grouped_explicit_identity_suite(
        editor, name, tpath2slot2group2terms, name, nsamples=nsamples)
    return s, n, data


def get_disability_suite(editor, group_key="GROUP", nsamples=None, name=None):
    df = editor.lexicons['disability']

    # only keep recommended
    df = df[df["TYPE"] != "nonrec"]
    pos_key = "POS"

    np_groups = defaultdict(list)
    adj_groups = defaultdict(list)
    pp_groups = defaultdict(list)

    for group in df[group_key].unique():
        if not group or type(group) != str:
            continue

        agroup = df[df[group_key] == group]
        group = group.lower()

        adjs = agroup[agroup[pos_key] == "adj"]
        fill_from_adjs(editor, list(adjs.index), group, np_groups, adj_groups)
        nouns = agroup[agroup[pos_key] == "n"]
        np_groups[group] += [term for term in nouns.index
                             if term and type(term) == str]
        pps = agroup[agroup[pos_key] == "pp"]
        fill_from_pps(editor, list(pps.index), group, np_groups, pp_groups)

    tpath2slot2group2terms = {
        cfg.disability_template_path: {
            "disability_np": np_groups,
            "disability_people": get_person_group(
                editor, 'people', pp_groups, adj_groups),
            "disability_friend": get_person_group(
                editor, 'friend', pp_groups, adj_groups),
            "disability_athletes": get_person_group(
                editor, 'athletes', pp_groups, adj_groups),
            "disability_student":  get_person_group(
                editor, 'student', pp_groups, adj_groups),
            "disability_liar": get_person_group(
                editor, 'liar', pp_groups, adj_groups),
            "disability_adj_or_pp": merge_groups(editor, pp_groups, adj_groups)
        },
        cfg.generic_template_path_adj_or_pp: {
            "identity_np": np_groups,
            "identity_writer": get_person_group(
                editor, 'writer', pp_groups, adj_groups),
            "identity_researcher": get_person_group(
                editor, 'researcher', pp_groups, adj_groups),
            "identity_individuals": get_person_group(
                editor, 'individuals', pp_groups, adj_groups),
            "identity_people":  get_person_group(
                editor, 'people', pp_groups, adj_groups),
            "identity_folk": get_person_group(
                editor, 'folk', pp_groups, adj_groups),
            "identity_activist": get_person_group(
                editor, 'activist', pp_groups, adj_groups),
            "identity_writers": get_person_group(
                editor, 'writers', pp_groups, adj_groups),
            "identity_lawyer": get_person_group(
                editor, 'lawyer', pp_groups, adj_groups),
            "identity_students": get_person_group(
                editor, 'students', pp_groups, adj_groups),
            "identity_character": get_person_group(
                editor, 'character', pp_groups, adj_groups),
        }
    }

    if not name:
        name = 'grouped_disability'
    s, n, data = create_grouped_explicit_identity_suite(
        editor, name, tpath2slot2group2terms, name, nsamples=nsamples)
    return s, n, data
