# adapted from https://github.com/allenai/allennlp/blob/master/allennlp/training/metrics/span_based_f1_measure.py

from typing import Dict, List, Optional, Set, Callable
from collections import defaultdict

from expanded_checklist.checklist.utils import tokenize, get_new_seq_labels

from allennlp.data.dataset_readers.dataset_utils.span_utils import (
    bio_tags_to_spans,
    bioul_tags_to_spans,
    iob1_tags_to_spans,
    bmes_tags_to_spans,
    TypedStringSpan,
)

TAGS_TO_SPANS_FUNCTION_TYPE =\
     Callable[[List[str], Optional[List[str]]], List[TypedStringSpan]]


def adjust_gold_label(glabel, meta_dict):
    """
    Adjust the gold label based on how many tokens were used to fill the ide-
    ntity slot.
    """
    # TODO: this is not error-proof (the context may influence tokenization
    # of the fill) + adjust for different labeling schemas
    fill_toks = tokenize(str(meta_dict['GROUP_FILL']))

    if len(fill_toks) <= 1:
        return glabel
    else:
        return get_new_seq_labels(
            meta_dict, glabel, n_fill_toks=len(fill_toks))


def calculate_span_based_f1_measure(
    predictions: List,
    gold_labels: List,
    meta: List,
    ignore_classes: List[str] = None,
    label_encoding: str = "BIOUL"
):
    if label_encoding not in ["BIO", "IOB1", "BIOUL", "BMES"]:
        raise Exception(
            "Unknown label encoding - expected 'BIO', 'IOB1', 'BIOUL', " +
            "'BMES'."
        )

    # These will hold per label span counts.
    true_positives: Dict[str, int] = defaultdict(int)
    false_positives: Dict[str, int] = defaultdict(int)
    false_negatives: Dict[str, int] = defaultdict(int)

    for i in range(len(gold_labels)):
        predicted_string_labels = predictions[i]
        gold_string_labels = gold_labels[i]
        gold_string_labels = adjust_gold_label(gold_string_labels, meta[i])

        tags_to_spans_function: TAGS_TO_SPANS_FUNCTION_TYPE
        if label_encoding == "BIO":
            tags_to_spans_function = bio_tags_to_spans
        elif label_encoding == "IOB1":
            tags_to_spans_function = iob1_tags_to_spans
        elif label_encoding == "BIOUL":
            tags_to_spans_function = bioul_tags_to_spans
        elif label_encoding == "BMES":
            tags_to_spans_function = bmes_tags_to_spans
        else:
            raise ValueError(
                f"Unexpected label encoding scheme '{label_encoding}'")

        predicted_spans = tags_to_spans_function(
            predicted_string_labels, ignore_classes)
        gold_spans = tags_to_spans_function(
            gold_string_labels, ignore_classes)

        for span in predicted_spans:
            if span in gold_spans:
                true_positives[span[0]] += 1
                gold_spans.remove(span)
            else:
                # NOTE: false positive is also the case where the span
                # is partially correct
                false_positives[span[0]] += 1
        # These spans weren't predicted.
        for span in gold_spans:
            false_negatives[span[0]] += 1

    all_tags: Set[str] = set()
    all_tags.update(true_positives.keys())
    all_tags.update(false_positives.keys())
    all_tags.update(false_negatives.keys())
    all_tags = sorted(all_tags)


    cl2res = {}
    for tag in all_tags:
        tag_metrics = compute_metrics(
            true_positives[tag],
            false_positives[tag],
            false_negatives[tag]
        )
        cl2res[tag] = tag_metrics

    # Compute the precision, recall and f1 for all spans jointly (micro)
    all_metrics = compute_metrics(
        sum(true_positives.values()),
        sum(false_positives.values()),
        sum(false_negatives.values())
    )
    cl2res['all'] = all_metrics
    return cl2res


def compute_metrics(
    true_positives: int, false_positives: int, false_negatives: int
):
    metrics = {}
    precision = true_positives / (true_positives + false_positives + 1e-13)
    recall = true_positives / (true_positives + false_negatives + 1e-13)
    fnr = false_negatives / (true_positives + false_negatives + 1e-13)
    f1_measure = 2.0 * (precision * recall) / (precision + recall + 1e-13)

    metrics["precision"] = precision
    metrics["recall"] = recall
    metrics["F1"] = f1_measure
    metrics["TP"] = true_positives
    metrics["FP"] = false_positives
    metrics["FN"] = false_negatives
    metrics["FNR"] = fnr
    metrics['TPR'] = metrics["recall"]
    return metrics
