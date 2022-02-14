from typing import List, Dict, Callable
from expanded_checklist.checklist.test_suite import TestSuite
from expanded_checklist.checklist.editor import Editor
from munch import Munch
import itertools
from collections import OrderedDict, namedtuple

from typing import Tuple, Collection

import src.config as cfg
import regex as re
import pandas as pd
import numpy as np

import spacy

from src.models.readers.read_data import read_data
import logging
from datasets import load_dataset

from expanded_checklist.checklist.tests import INV, Wilcoxon, PairedTtest, \
    PairedPermutation, Permutation, BasicClassificationMetrics, \
    EqualityDifference, SubgroupMetrics, CounterfactualGap, ProbDiffStats

# use the same tokenization throughout the project
from expanded_checklist.checklist.utils import tokenize, expand_label

from copy import deepcopy

logging.basicConfig(level=cfg.log_level)
logger = logging.getLogger(__name__)

NamedRegex = namedtuple(
    'NamedRegex', ['basic', 'caps', 'abasic', 'acaps'])


#############################
#   TEMPLATE CREATION       #
#############################

def _get_entity_queue(entity_str):
    # TODO: adjust to different labeling schemes
    # (for now only BIOUL is supported)
    entity_queue = entity_str.split('|')

    # [[person], B-PER]] [[last_name], L-PER]] etc.
    entity_queue = [x.split(":") for x in entity_queue if ":" in x]

    # if the labels were defined over the fill-slots and a slot was filled
    # with a multi-word experssion -- expand that slot to more slots
    new_entity_queue = []
    for e in entity_queue:
        surface_str, lab = e

        # TODO: for now, this is based on an assumption that the 'fillers'
        # are tokenised in the same way in and out of context
        toks = tokenize(surface_str)
        if len(toks) > 1:
            new_labels = expand_label(lab, len(toks))
            for t, l in zip(toks, new_labels):
                new_entity_queue.append([t, l])
        else:
            new_entity_queue.append(e)
    return new_entity_queue


def _get_label(template_dict, task):
    # the templates support only those 3 classes for classification for now
    sent_class2int = {"neut": 1, "neg": 0, "pos": 2, "0": 0, "1": 1, "2": 2}

    if task == "SENT":
        if "SENT" not in template_dict or not template_dict["SENT"]:
            return None
        label = template_dict["SENT"]

        if not isinstance(label, int):
            label = sent_class2int.setdefault(label, None)
        return label
    elif task == "NER":
        # go index by index in the tok sentence and if tok is the same as
        # the top NER entity, pop and assign that entity label, else assign O.
        if "NER" not in template_dict or not template_dict["NER"]:
            return None

        tok_temp_sent = template_dict["TOKENIZED_TEMPLATE"]

        # establish the base labels for the original template tokens
        # -- we handle multi-word expressions for identity terms when labels
        # are compared
        entity_queue = _get_entity_queue(template_dict["NER"])

        label = ["O"] * len(tok_temp_sent)
        for i, tok in enumerate(tok_temp_sent):
            if not entity_queue:
                break
            next_entity_tok, next_entity_lab = entity_queue[0]
            if str(tok) == next_entity_tok:
                label[i] = next_entity_lab
                entity_queue.pop(0)

        if len(entity_queue) > 0:
            raise Exception('Not all entities got assigned tokens!')
        return label
    else:
        raise Exception(
            f'Unsupported task {task} detected in template processing')


def _get_regex(identity_key, ind_chars=("{", "}")):
    lc, rc = ind_chars
    reg = fr'{lc}({identity_key}){rc}'
    reg2 = fr'{lc}(a:{identity_key}){rc}'
    reg_caps = fr'{lc}({identity_key[0].upper() + identity_key[1:]}){rc}'
    reg_caps2 = fr'{lc}(a:{identity_key[0].upper() + identity_key[1:]}){rc}'
    return NamedRegex(reg, reg_caps, reg2, reg_caps2)


def _replace_key_indicators(
        s, identity_key, ind_chars=("{", "}"), new_ind_chars=("@", "@")):
    named_regx = _get_regex(identity_key, ind_chars)
    lc, rc = new_ind_chars

    for reg in named_regx:
        s = re.sub(reg, fr"{lc}\1{rc}", s)
    return s


def _expand_non_identity_terms(
    editor: Editor,
    templates: List[OrderedDict],
    non_indentity_samples: int,
    identity_key: str
) -> List[OrderedDict]:
    """
    Expand all the templated sentences that have a non-identity word
    to be sampled. Keep all the original keys as they are, only sample
    non-identity terms here.
    """
    # change the key indicators for identity keys so that they are kept
    # as they are
    for t in templates:
        t['TEMPLATE'] = _replace_key_indicators(
            t['TEMPLATE'], identity_key, new_ind_chars=("@", "@"))
        t['NER'] = _replace_key_indicators(
            t['NER'], identity_key, new_ind_chars=("@", "@"))

    t = editor.template(
        templates,
        nsamples=non_indentity_samples,
        unroll=True
    )

    temps_in = set()
    to_ret = []
    for x in t.data:
        if x['TEMPLATE'] not in temps_in:
            to_ret.append(x)
            temps_in.add(x['TEMPLATE'])

    to_ret = sorted(to_ret, key=lambda x: x['TEMPLATE'])

    for t in to_ret:
        t['TEMPLATE'] = _replace_key_indicators(
            t['TEMPLATE'], identity_key,
            ind_chars=("@", "@"), new_ind_chars=("{", "}"))
        t['NER'] = _replace_key_indicators(
            t['NER'], identity_key,
            ind_chars=("@", "@"), new_ind_chars=("{", "}"))
    return to_ret


def _get_initialized_labels_dict(task2n_classes):
    # we keep a label dictionary -- to support labels for different tasks
    labels_dict = {}
    for x in cfg.supported_tasks:
        if task2n_classes is not None:
            n_classes = task2n_classes[x]
        # those kinds of classes are assumed in the templates
        else:
            if x == "SENT":
                n_classes = 3
            elif x == "NER":
                n_classes = 17
            else:
                n_classes = None
        labels_dict[x] = Munch({"labels": [], "n_classes": n_classes})
    return labels_dict


def _prevent_key_tokenization(template_dict, identity_key):
    # change {} to @@ -- to prevent the keys from tokenization
    template_dict["TEMPLATE"] = _replace_key_indicators(
        template_dict["TEMPLATE"], identity_key, new_ind_chars=("@", "@"))

    if "NER" in template_dict and template_dict["NER"]:
        template_dict["NER"] = _replace_key_indicators(
            template_dict["NER"], identity_key, new_ind_chars=("@", "@"))


def _substitute_ik(identity_key, fill, tslot):
    regs_tuple = _get_regex(identity_key, ind_chars=("{", "}"))
    tslot = re.sub(regs_tuple.basic, "{" + fill + "}", tslot)
    tslot = re.sub(regs_tuple.abasic, "{a:" + fill + "}", tslot)
    tslot = re.sub(
        regs_tuple.caps, "{" + fill[0].upper() + fill[1:] + "}", tslot)
    tslot = re.sub(
        regs_tuple.acaps, "{a:" + fill[0].upper() + fill[1:] + "}", tslot)
    return tslot


def _check_only_given_slots_missing(editor, identity_key, template_slot):
    regs_tuple = _get_regex(identity_key)
    if not any(re.search(reg, template_slot) for reg in regs_tuple):
        return False
    tmp = template_slot
    for reg in regs_tuple:
        tmp = re.sub(reg, "x", tmp)
    return _all_keys_in_lexicon_or_args(editor, tmp)


def _all_keys_in_lexicon_or_args(
    editor: Editor,
    sent: str,
    key_args: Collection[str] = []
) -> bool:
    keys = re.findall(r"{[^}]+}", sent)
    sent_missing_keys = set()

    for key in keys:
        # strip {}
        clean_key = key[1:-1]
        # keys in lexicons etc. should start with lowercase
        clean_key = clean_key[0].lower() + clean_key[1:]
        if clean_key in editor.lexicons or clean_key in key_args:
            continue
        # rely on the inner checklist tests (for adding dets etc.)
        try:
            kwargs = {k: ['tmp'] for k in key_args}
            editor.template(key, **kwargs)
        except Exception:
            sent_missing_keys.add(key)
    if len(sent_missing_keys) > 0:
        return False
    else:
        return True


def get_tokenized_template(template_dict):
    cleaned_tmp = re.sub(
        r"@a:([a-zA-Z0-9_]+)@", r"a @\g<1>@", template_dict["TEMPLATE"])
    tmp_toks = tokenize(cleaned_tmp)
    tok_temp_sent = []
    for tok in tmp_toks:
        m = re.match(r"(@[a-zA-Z0-9_:]+@)(.+)", tok)
        if m:
            # the second group is likely to be a punctuation symbol
            tok_temp_sent.append(m.group(1))
            tok_temp_sent.append(m.group(2))
        else:
            tok_temp_sent.append(tok)
    template_dict["TOKENIZED_TEMPLATE"] = tok_temp_sent


def get_grouped_data_from_template(
    editor: Editor,
    df_templates: pd.DataFrame,
    groups: Dict[str, List],  # {"male_names":["Max", "Daniel"]} etc.
    identity_key: str,
    nsamples: int = None,
    term_property: str = "",
    # if groups contain Munch objs, specify property that points to the term
    new_keys: List[str] = None,
    unroll: bool = True,
    # the non-identity terms are sampled first
    non_indentity_samples: int = 3,  # e.g. positive adjs etc.
    task2n_classes: Dict = None,
    **kwargs
) -> Tuple[List, List]:
    """
    Can used to create either:
    (i) semantically paired examples, by specifying new_keys to map to
        different properties of the same new_key e.g.
        [new_key.term1, new_key.term2], in such case groups should be set to
        groups={'new_key': List_of_munch_objs} and term_property should be ""
    (ii) non-semantically paired examples, by leaving new_keys as None
        and specyfing multiple groups, e.g.
        groups={'male_names': List_of_munch/str,
            'female_names': List_of_munch/str}
        If the the groups correspond to list of munches then one should
        specify the 'term_property' of those munches (property that points
        to the actual str term) -- this property will be used to fill the
        templates while the remaining info in the munch will go to metadata

    The function handles all the metadata -- which can be later used to
    return more useful statistics/info for the test and allows for extra
    metrics to be run on top of the test
    """
    templates: List[OrderedDict] = df_templates.to_dict('records')
    templates = sorted(templates, key=lambda x: x['TEMPLATE'])
    templates = [t for t in templates if _check_only_given_slots_missing(
                    editor, identity_key, t['TEMPLATE'])]

    templates = _expand_non_identity_terms(
        editor, templates, non_indentity_samples, identity_key)

    labels_dict = _get_initialized_labels_dict(task2n_classes)
    data, meta = [], []

    # keep the samples the same across all templates
    if nsamples is not None:
        for g, options in groups.items():
            if nsamples >= len(options):
                continue
            groups[g] = list(
                np.random.choice(options, size=nsamples, replace=False))

    for x in templates:
        template_slot = x['TEMPLATE']
        x["IDENTITY_KEY"] = identity_key

        _prevent_key_tokenization(x, identity_key)
        get_tokenized_template(x)

        # semantically grouped
        if new_keys:
            assert len(groups) == 1

            # get labels: same for all groups
            for task in cfg.supported_tasks:
                label = _get_label(x, task)
                # each sentence template yields as many versions
                # as the number of groups
                labels_dict[task].labels +=\
                    [label] * len(list(groups.values())[0])

            # new keys are e.g. person.female, person.male etc.
            new_keys = sorted(new_keys)
            group_names = [x.split(".", 1)[1] for x in new_keys]
            template_slots = [_substitute_ik(
                identity_key, new_key, template_slot) for new_key in new_keys]

            t = editor.template(
                    template_slots,
                    nsamples=None,
                    meta=True,
                    remove_duplicates=True,
                    **groups,
                    **kwargs)

            data += t.data
            new_meta = []
            for n, m in enumerate(t.meta):
                example_meta = deepcopy(x)
                example_meta['SAMPLE'] = {}
                for i, gname in enumerate(group_names):
                    example_meta[i] = gname
                    example_meta['SAMPLE'][gname] = m[identity_key][gname]
                new_meta.append(example_meta)
            meta += new_meta

        # not semantically grouped
        else:
            # get labels: same for all groups
            for task in cfg.supported_tasks:
                label = _get_label(x, task)
                labels_dict[task].labels.append(label)

            # meta holds all fields that the original template dict
            # e.g. info about domain, label etc.
            example_meta = x
            example_meta['SAMPLE'] = {}
            example_data = []

            # get data for all groups from that tempate
            if term_property:
                new_key = f"{identity_key}.{term_property}"
                template_slot = _substitute_ik(
                    identity_key, new_key, template_slot)

            group_names = sorted(list(groups.keys()))
            for i, gname in enumerate(group_names):
                group_kwargs = {identity_key: groups[gname]}
                t = editor.template(
                    template_slot,
                    nsamples=None,
                    meta=True,
                    remove_duplicates=True,
                    **group_kwargs,
                    **kwargs)

                # each group has a list of sentences for each template
                example_data.append(t.data)
                example_meta[i] = gname

                example_meta['SAMPLE'][gname] = []
                for m in t.meta:
                    if identity_key in m:
                        example_meta['SAMPLE'][gname].append(m[identity_key])
                    else:
                        example_meta['SAMPLE'][gname].append(
                            m[identity_key[0].upper() + identity_key[1:]]
                        )

            data.append(example_data)
            meta.append(example_meta)

    return data, meta, labels_dict
