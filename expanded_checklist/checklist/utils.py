import dill
import numpy as np
from typing import Tuple, List
import itertools
import regex as re
import json
from enum import Enum
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer

import logging
logger = logging.getLogger(__name__)

sequence_tasks = ["NER"]

# tokenizer is used for sequence labeling data
tokenizer = SpacyTokenizer(language="en_core_web_sm", pos_tags=True)

ACCUMULATED_STR = "accumulated"
TOKENIZATION_DICT = "TOKENIZATION_DICT"

# max of used groupings when creating samples for counterfactual tests
# (if FlattenGroup is RANDOM_MATCH)
MAX_N_GROUPINGS = 100


def tokenize(sent: str) -> List[str]:
    return [x.text for x in tokenizer.tokenize(sent)]


class FlattenGroup(Enum):
    RANDOM_MATCH = 0
    AVERAGE = 1
    NONE = 2
    FLATTEN = 3
    FLATTEN_ALL = 4


class DataShape(Enum):
    GROUPED = 1,    # #ngroups x #nexamples
    UNGROUPED = 2   # #nexamples x #ngroups


def load_test(file):
    dill._dill._reverse_typemap['ClassType'] = type
    return dill.load(open(file, 'rb'))


def read_pred_file(
    path, file_format=None, format_fn=None, ignore_header=False
) -> Tuple[List, List]:
    preds = []
    confs = []
    if file_format is None and format_fn is None:
        file_format = 'pred_and_softmax'
    if file_format == 'pred_only':
        format_fn = lambda x: (x, 1)
    elif file_format == 'binary_conf':
        def formatz(x):
            conf = float(x)
            confs = np.array([1 - conf, conf])
            pred = int(np.argmax(confs))
            return pred, confs
        format_fn = formatz
    elif file_format == 'softmax':
        def formatz(x):
            confs = np.array([float(y) for y in x.split()])
            pred = int(np.argmax(confs))
            return pred, confs
        format_fn = formatz
    elif file_format == 'pred_and_conf':
        def formatz(x):
            pred, conf = x.split()
            if pred.isdigit():
                pred = int(pred)
            return pred, float(conf)
        format_fn = formatz
    elif file_format == 'pred_and_softmax':
        def formatz(x):
            allz = x.split()
            pred = allz[0]
            confs = np.array([float(x) for x in allz[1:]])
            if pred.isdigit():
                pred = int(pred)
            return pred, confs
        format_fn = formatz
    elif file_format == "seq_pred_and_softmax":
        def formatz(x):
            # list of tags, list of probs, list of label vocab (ordered based
            # on the index)
            m = re.match(r"(\[[^\]\[]+\]) (\[.+\])", x)  # (\[.+\])", x)
            if not m:
                raise(Exception("Incorrect format of predictions for " +
                                "sequence labeling."))
            pred = m.group(1)
            confs = m.group(2)
            # words = m.group(3)

            # a list of predicted classes
            pred = re.sub("'", '"', pred)
            pred = json.loads(pred)
            # 2d array: list of softmax outputs
            confs = np.array(json.loads(confs), dtype=float)

            return pred, confs
        format_fn = formatz
    elif file_format is None:
        pass
    else:
        raise(Exception(
            'file_format %s not suported. Accepted values are pred_only, ' +
            'softmax, binary_conf, pred_and_conf, pred_and_softmax'
            % file_format))

    with open(path, 'r') as f:
        if ignore_header:
            f.readline()
        label_vocab = None
        for i, line in enumerate(f):
            if i == 0:
                # first line may contain information about label vocab
                m = re.match(r"(\[[^\]\[]+\])", line.strip())
                if m:
                    line = line.strip()
                    line = re.sub("'", '"', line)
                    label_vocab = json.loads(line)

            if i != 0 or label_vocab is None:
                line = line.strip('\n')
                p, c = format_fn(line)
                preds.append(p)
                confs.append(c)

    if file_format == 'pred_only' and all([x.isdigit() for x in preds]):
        preds = [int(x) for x in preds]

    if not label_vocab:
        return preds, confs
    else:
        return preds, confs, label_vocab


def iter_with_optional(data, preds, confs, labels, meta, idxs=None):
    # If this is a single example
    if type(data) not in [list, np.array]:
        return [(data, preds, confs, labels, meta)]
    if type(meta) not in [list, np.array]:
        meta = itertools.repeat(meta)
    else:
        if len(meta) != len(data):
            raise(Exception('If meta is list, length must match data'))
    if type(labels) not in [list, np.array]:
        labels = itertools.repeat(labels)
    elif len(labels) != len(data):
        # a label for a single example can be a list for seq tasks
        labels = itertools.repeat(labels)

    ret = zip(data, preds, confs, labels, meta)
    if idxs is not None:
        ret = list(ret)
        ret = [ret[i] for i in idxs]
    return ret


#####################################
#      RESTRUCTURING THE DATA       #
#####################################

def flatten_confs_and_labels(
        confs, labels, data_structure, skip_non_labeled=False):
    """
    Can be used for sequence labelling. Each token becomes a separate example.
    This should only be called on sequence data.

    Note: this does not take into account the potential mismatches in tokens
    for different groups (!). Those have to be handled separately.
    """
    if data_structure != DataShape.GROUPED:
        raise Exception('Cannot flatten ungrouped data!')

    new_confs = []
    for i, group_data in enumerate(confs):
        new_group_data = []
        for j, sent_data in enumerate(group_data):
            if labels[i][j] is None and skip_non_labeled:
                continue

            for token_data in sent_data:
                new_group_data.append(token_data)
        new_confs.append(new_group_data)

    new_labels = []
    for i, group_data in enumerate(labels):
        new_group_data = []
        for j, sent_data in enumerate(group_data):
            if sent_data is None:
                if skip_non_labeled:
                    continue
                else:
                    sent_data = [None] * len(confs[i][j])

            for token_data in sent_data:
                new_group_data.append(token_data)
        new_labels.append(new_group_data)

    return new_confs, new_labels


##########################
#      DATA CHECKS       #
##########################

def is_2d_list(data) -> bool:
    ltypes = [list, np.ndarray]

    if type(data) in ltypes and len(data) > 0 and \
            type(data[0]) in ltypes and len(data[0]) > 0 and \
            type(data[0][0]) not in ltypes:
        return True
    else:
        return False


def is_1d_list(data) -> bool:
    ltypes = [list, np.ndarray]

    if type(data) in ltypes and (
            (len(data) > 0 and type(data[0]) not in ltypes) or len(data) == 0):
        return True
    else:
        return False


####################################################
#      HANDLING MISMATCHES IN TOKENS BETWEEN       #
#       LABELS AND PREDICTIONS (SEQ LABELING)      #
####################################################

def get_new_seq_labels(meta_dict, glabel, n_fill_toks=None, group_idx=None):
    new_labels = []

    if "TOKENIZED_TEMPLATE" in meta_dict:
        template_toks = meta_dict['TOKENIZED_TEMPLATE']
    else:
        template_toks = tokenize(meta_dict['TEMPLATE'])
    ikey = meta_dict['IDENTITY_KEY']

    if not n_fill_toks:
        gname = meta_dict[group_idx]
        if "." in gname:
            prop, term = gname.split(".")
            gfill = str(meta_dict['SAMPLE'][prop][term])
        else:
            gfill = str(meta_dict['SAMPLE'][gname])
        fill_toks = tokenize(gfill)
        n_fill_toks = len(fill_toks)

    for tok, label in zip(template_toks, glabel):
        if tok == f"@{ikey}@":
            new_labels += expand_label(label, n_fill_toks)
        else:
            new_labels.append(label)
    return new_labels


def expand_label(lab, n_toks):
    """
    Expand the label to cover the n_toks number of tokens.
    """
    # if labelsis O (or other like it) -- expand to O+
    if len(lab) < 2:
        new_labels = [lab] * n_toks
    else:
        # drop B-, I- etc.
        clean_label = get_class_from_seq_label(lab)

        if clean_label == lab:
            first, mid, last = "", "", ""
        elif lab[:2] == "U-":
            first, mid, last = "B-", "I-", "L-"
        elif lab[:2] == "B-":
            first, mid, last = "B-", "I-", "I-"
        elif lab[:2] == "L-":
            first, mid, last = "I-", "I-", "L-"
        elif lab[:2] == "I-":
            first, mid, last = "I-", "I-", "I-"

        new_labels = []
        for i in range(n_toks):
            if i == 0:
                new_labels.append(f"{first}{clean_label}")
            elif i == n_toks - 1:
                new_labels.append(f"{last}{clean_label}")
            else:
                new_labels.append(f"{mid}{clean_label}")
    return new_labels


def compare_preds_with_mismatched_tokens(pred1, pred2, meta_dict):
    if len(pred1) == len(pred2):
        return pred1 == pred2

    if "TOKENIZED_TEMPLATE" in meta_dict:
        template_toks = meta_dict['TOKENIZED_TEMPLATE']
    else:
        template_toks = tokenize(meta_dict['TEMPLATE'])
    n_temp_toks = len(template_toks)
    ikey = meta_dict['IDENTITY_KEY']

    n_filled_toks1 = len(pred1) - n_temp_toks + 1
    n_filled_toks2 = len(pred2) - n_temp_toks + 1

    p1, p2 = 0, 0
    for tok in template_toks:
        if tok == f"@{ikey}@":
            labels1 = set()
            labels2 = set()

            for i in range(p1, p1 + n_filled_toks1):
                pred = pred1[i]
                labels1.add(get_class_from_seq_label(pred))

            for i in range(p2, p2 + n_filled_toks2):
                pred = pred2[i]
                labels2.add(get_class_from_seq_label(pred))

            if len(labels1) > 1 or len(labels2) > 1 or labels1 != labels2:
                return False
            p1 += n_filled_toks1
            p2 += n_filled_toks2
        elif pred1[p1] != pred2[p2]:
            return False
        else:
            p1 += 1
            p2 += 1
    return True


####################
#      OTHER       #
####################

def convert_all_preds_into_ints(preds, label_vocab):
    """
    The function works for both GROUPED and UNGROUPED data shapes
    """
    new_preds = []
    for sent_preds in preds:
        if is_2d_list(sent_preds):
            tmp_list = []
            for sent_pred in sent_preds:
                # get indexes for the classes
                tmp_list.append(
                    np.array([label_vocab.index(x) for x in sent_pred]))
            new_preds.append(tmp_list)
        else:
            new_preds.append(
                np.array([label_vocab.index(x) for x in sent_preds]))
    return new_preds


def convert_all_labels_into_str(labels, data_structure):
    def covert_one_label_set(to_convert):
        converted = []
        if is_2d_list(to_convert):
            tmp_list = []
            for sent_labels in to_convert:
                tmp_list.append([str(x) for x in sent_labels])
            converted.append(tmp_list)
        else:
            converted = [str(x) for x in to_convert]
        return converted

    # one set of labels for all examples
    if data_structure == DataShape.UNGROUPED:
        new_labels = covert_one_label_set(labels)
    # different labels for different groups
    else:
        new_labels = []
        for group_labels in labels:
            new_labels.append(covert_one_label_set(group_labels))
    return new_labels


def convert_class_data_to_seq_data(labels, preds, confs, data_structure):
    """
    Data can be confs, preds, labels.

    Turn classification into seq labeling with sequences of length 1.
    This is used to have some functions compatible for both classification
    tasks and sequence labeling tasks.
    """
    def helper(data):
        new_data = []
        if data_structure == DataShape.GROUPED:
            for gdata in data:
                new_group_data = []
                for sent_data in gdata:
                    new_group_data.append([sent_data])
                new_data.append(new_group_data)
        else:
            for sent_data in data:
                new_sent_data = []
                if type(sent_data) not in [list, np.ndarray]:
                    new_sent_data.append(sent_data)
                else:
                    for gdata in sent_data:
                        new_sent_data.append([gdata])
                new_data.append(new_sent_data)
        return new_data

    return helper(labels), helper(preds), helper(confs)


def get_class_from_seq_label(lab):
    if lab == "O" or len(lab) < 2 or lab[1] != "-" or not lab[0].isupper():
        return lab
    else:
        return lab[2:]
