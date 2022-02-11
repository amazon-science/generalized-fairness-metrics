# Script used to process output from allennlp predict so that it can be
# processed by an evalaution core

import os
import sys
import argparse
import json
from functools import partial
import logging
from typing import List, Dict
import numpy as np
import json
import regex as re

logger = logging.getLogger(__name__)

# FROM CHECKLIST NOTEBOOKS:
# def pred_and_conf(data):
#     # change format to softmax, make everything in [0.33, 0.66] range be predicted as neutral
#     preds = batch_predict(model, data)
#     pr = np.array([x['score'] if x['label'] == 'POSITIVE' else 1 - x['score'] for x in preds])
#     pp = np.zeros((pr.shape[0], 3))
#     margin_neutral = 1/3.
#     mn = margin_neutral / 2.
#     neg = pr < 0.5 - mn
#     pp[neg, 0] = 1 - pr[neg]
#     pp[neg, 2] = pr[neg]
#     pos = pr > 0.5 + mn
#     pp[pos, 0] = 1 - pr[pos]
#     pp[pos, 2] = pr[pos]
#     neutral_pos = (pr >= 0.5) * (pr < 0.5 + mn)
#     pp[neutral_pos, 1] = 1 - (1 / margin_neutral) * np.abs(pr[neutral_pos] - 0.5)
#     pp[neutral_pos, 2] = 1 - pp[neutral_pos, 1]
#     neutral_neg = (pr < 0.5) * (pr > 0.5 - mn)
#     pp[neutral_neg, 1] = 1 - (1 / margin_neutral) * np.abs(pr[neutral_neg] - 0.5)
#     pp[neutral_neg, 0] = 1 - pp[neutral_neg, 1]
#     preds = np.argmax(pp, axis=1)
#     return preds, pp


def softmax(vec):
    nom = [np.exp(y) for y in vec]
    den = sum(nom)
    return [x/den for x in nom]


def convert_line(json_line: str, **kwargs):
    """
    This function recognised whether the predictions 'belong' to a seq tagging
    model or a text classification model (and gets the appropriate info out).
    """
    json_line = json_line.strip()

    try:
        json_dict = json.loads(json_line)
    except Exception:
        return ""

    if 'tags' in json_dict:
        return convert_line_seq(json_dict, **kwargs)
    elif 'label' in json_dict:
        return convert_line_classification(json_dict, **kwargs)


def convert_line_seq(
    json_dict: Dict,
    **kwargs
 ) -> str:
    """
    This is for sequences so lists of logits are 2d
    """
    def mround(match):
        return "{:.4f}".format(float(match.group()))

    logits = json_dict['logits']

    # list of str
    tags = json_dict['tags']
    probs = [softmax(x) for x in logits]
    # words = json_dict['words']

    tags = json.dumps(tags)
    probs = json.dumps(probs)

    # find numbers with 4 or more digits after the decimal point
    probs = re.sub(r"\d+\.\d{4,}", mround, probs)

    return " ".join([tags, probs]) + "\n"


def convert_line_classification(
    json_dict: Dict,
    reordering_indices: List[int],
    add_neutral_class: bool = False,
    **kwargs
 ) -> str:
    probs = json_dict['probs']
    label = str(json_dict['label'])

    # order probabilities for the labels
    # first is the prob for label '0', then for '1' etc.
    probs = np.array(probs)[reordering_indices]

    # follow checklist procedure (based on the code above)
    if add_neutral_class and len(probs) == 2:
        if probs[0] > 1/3 and probs[0] < 2/3:
            # probability of positive review
            if label == "1":
                pr = max(probs)
            else:
                pr = 1 - max(probs)

            pp = np.zeros(3)
            margin_neutral = 1/3.
            mn = margin_neutral / 2.

            # shift the probability mass
            neutral_pos = (pr >= 0.5) * (pr < 0.5 + mn)
            if neutral_pos:
                pp[1] = 1 - (1 / margin_neutral) * np.abs(pr - 0.5)
                pp[2] = 1 - pp[1]

            neutral_neg = (pr < 0.5) * (pr > 0.5 - mn)
            if neutral_neg:
                pp[1] = 1 - (1 / margin_neutral) * np.abs(pr - 0.5)
                pp[0] = 1 - pp[1]

            probs = pp
            label = "1"
        else:
            probs = [probs[0], 0, probs[1]]
            if label == "1":
                label = "2"

    line_vals = [str(label)] + [f"{x:0.4f}" for x in probs]
    return " ".join(line_vals) + "\n"


def get_prob_reordering(
    labels: List[str],
) -> List[int]:
    reordering = np.zeros(len(labels), dtype=int)

    err = False
    for i, label in enumerate(labels):
        try:
            label_index = int(label)
            reordering[label_index] = i
        except Exception:
            err = True
    if err:
        logger.warning(
            "Couldn't get label reordering. This is ok if the" +
            "predictions are *not* for a classification model.")
    return reordering, not err


def get_labels(
    labels_vocab_path: str
) -> List[str]:
    with open(labels_vocab_path, "r") as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines if x]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-path', type=str)
    parser.add_argument('--labels-vocab-path', type=str)
    parser.add_argument('--out-path', type=str)
    parser.add_argument('--add-neutral-class', action='store_true')
    args = parser.parse_args()

    if not os.path.isfile(args.in_path):
        logger.error(f"File {args.in_path} does not exist.")
        sys.exit()

    if not os.path.exists(os.path.dirname(args.out_path)):
        dirname = os.path.dirname(args.out_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

    with open(args.in_path, "r") as f:
        json_lines = f.read().split("\n")

        labels = get_labels(args.labels_vocab_path)

        # to map softmax output to the labels
        reordering_indices, succ = get_prob_reordering(labels)
        if succ:
            new_labels = ["" for i in range(len(labels))]
            for new_index, old_index in enumerate(reordering_indices):
                new_labels[new_index] = labels[old_index]
            labels = new_labels

        convert_fun = partial(
            convert_line,
            labels=labels,
            reordering_indices=reordering_indices,
            add_neutral_class=args.add_neutral_class)

        converted_lines = map(convert_fun, json_lines)
        with open(args.out_path, "w") as f2:
            f2.write(f"{str(labels)}\n")
            f2.writelines(converted_lines)


if __name__ == "__main__":
    main()
