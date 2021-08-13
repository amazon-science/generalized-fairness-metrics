import dill
from munch import Munch
import numpy as np
import os

from .core_record import CoreRecord
from .utils import load_test, read_pred_file, tokenize, \
    sequence_tasks, DataShape, is_1d_list, is_2d_list, ACCUMULATED_STR, \
    TOKENIZATION_DICT
from .tests import AbstractTest, BasicClassificationMetrics, BasicSeqMetrics

from typing import Dict, List

from enum import Enum
from collections import namedtuple, defaultdict

import logging
from copy import deepcopy
logger = logging.getLogger(__name__)

# EXAMPLE OF A METADATA:
# {'TEMPLATE': 'As a child, @person@ had big dreams.', 'DOMAIN': 'personal',
# 'CLASS': 'neut', 'EXTENDABLE': '', 'CORPUS': '', 'IDENTITY_KEY': 'person',
# 0: 'white', 1: 'other_race', 'SAMPLE': {'scientists': 'phonetician',
# 'pos_adj_event': 'wonderful', 'neg_adj_event': 'frightening', 'neg_v_2b_vbd':
# 'despised', 'other_race': 'Anna, who is latino', 'white': 'Anna, who is
# white', 'neg_adj_personstate': 'horrified', 'neg_anger_adj_personstate':
# 'llivid', 'kiritchenko_emotional_state': 'disappointed', 'pos_v_2b_vbd':
# 'loved', 'kiritchenko_emotional_situation': 'depressing'}, 'SAMPLE_n': 0}


CacheEntry = namedtuple(
    'CacheEntry',
    ['data_filter', 'data_shape', 'prob_based',
     'drop_none_labels', 'flatten_method'])


class EvaluationCore():
    def __init__(
        self, data, labels_dict, meta=None,
        name=None, capability=None,
        description=None, tests=None,
        non_matched_data=False,
        group_names=None
    ):
        """
        A class which is used to store the data, metadata and labels.
        It can also load model's predictions for the data it holds and
        'tie' it to the data.

        It is equivalent to the checklist test, modulo all test-specific
        functionality (running specific tests, getting results etc.) and
        reuses a lot of original checklist code.

        This class is a result of decoupling the data and prediction management
        from getting the results for a specific type of test.

        TODO: subsampling doesn't work for now.

        Parameters
        ----------
        data : list
            List or list(lists) of whatever the model takes as input.
            Strings, tuples, etc.
        labels_dict : a dictionary mapping a name of the task to:
            a Munch with nclasses field and labels field:
                a single value (int, str, etc) or list
                If list, must be the same length as data
        meta : list
            metadata for examples, must be the same length as data
        name : str
            test name
        capability : str
            test capability
        description : str
            test description
        non_matched_data: bool
            Indicates whether the data is 'counterfactual': many different
            version of the sentence, one for each group (True) or 'non-matched'
            : each group has its own set of sentences and labels.
        group_names: List[str]
            Provide when non_matched_data is True.
        """
        self.data = data

        self.labels_dict = labels_dict
        self.labels = None

        if non_matched_data:
            self.data_structure = DataShape.GROUPED
            self.group_names = group_names
        else:
            self.data_structure = DataShape.UNGROUPED
            self.group_names = None

        self.meta = meta
        self.run_idxs = None
        self.result_indexes = None
        self.name = name
        self.capability = capability
        self.description = description
        self.tests = tests
        self.label_vocab = None
        self.results = None

        # cache of core records processed in different ways
        self.core_record_cache = {}

    def set_tests(self, tests):
        """
        Set the tests/metrics that should be used for evaluation.
        """
        self.tests = tests

    def save(self, file):
        with open(file, 'wb') as f:
            dill.dump(self, f, recurse=True)

    @staticmethod
    def from_file(file):
        return load_test(file)

    def example_list_and_indices(self, n=None, seed=None):
        """Subsamples test cases

        Parameters
        ----------
        n : int
            Number of testcases to sample
        seed : int
            Seed to use

        Returns
        -------
        tuple(list, list)
            First list is a list of examples
            Second list maps examples to testcases.

        For example, let's say we have two testcases: [a, b, c] and [d, e].
        The first list will be [a, b, c, d, e]
        the second list will be [0, 0, 0, 1, 1]

        Also updates self.run_idxs if n is not None to indicate which testcases
        were run. Also updates self.result_indexes with the second list.

        """
        if seed is not None:
            np.random.seed(seed)
        self.run_idxs = None
        idxs = list(range(len(self.data)))
        if n is not None:
            idxs = np.random.choice(idxs, min(n, len(idxs)), replace=False)
            self.run_idxs = idxs

        if is_1d_list(self.data[0]):
            all_data = [
                (i, y, m) for i in idxs
                for (y, m) in zip(self.data[i], self.meta[i])]
            result_indexes, examples, meta = map(list, list(zip(*all_data)))
        # e.g. for each tempalte there are many groups and each group
        # can be represented with many terms
        elif is_2d_list(self.data[0]):
            all_data = []
            for i in idxs:
                example_data = self.data[i]
                example_meta = self.meta[i]

                for group_idx, sent_list in enumerate(example_data):
                    for y in sent_list:
                        all_data.append(((i, group_idx), y, example_meta))
            result_indexes, examples, meta = map(list, list(zip(*all_data)))
        else:
            examples = [self.data[i] for i in idxs]
            meta = [self.meta[i] for i in idxs]
            result_indexes = idxs  # list(range(len(self.data)))

        self.result_indexes = result_indexes
        return examples, meta, result_indexes

    # def recover_example_list_and_indices(self):
    #     """Recovers a previously computed example_list_and_indices"""
    #     idxs = list(range(len(self.data)))
    #     if self.run_idxs is not None:
    #         idxs = self.run_idxs
    #     if is_1d_list(self.data[0]):
    #         examples = [y for i in idxs for y in self.data[i]]
    #         meta = [y for i in idxs for y in self.meta[i]]
    #     elif is_2d_list(self.data[0]):
    #         pass
    #         # TODO
    #     else:
    #         examples = [self.data[i] for i in idxs]
    #     result_indexes = self.result_indexes
    #     return examples, result_indexes

    def update_results_from_preds(self, preds, confs):
        """Updates results from preds and confs
        Assumes that example_lists_and_indices or to_raw_examples or
        to_raw_file was called before, so that self.result_indexes exists
        Parameters
        ----------
        preds : list
            Predictions
        confs : list
            Confidences

        Updates self.results.preds and self.results.confs
        """
        result_indexes = self.result_indexes
        if is_1d_list(self.data[0]):
            self.results.preds = [[] for _ in self.data]
            self.results.confs = [[] for _ in self.data]
            for i, p, c in zip(result_indexes, preds, confs):
                self.results.preds[i].append(p)
                self.results.confs[i].append(c)
        elif is_2d_list(self.data[0]):
            self.results.preds = [[[] for _ in x] for x in self.data]
            self.results.confs = [[[] for _ in x] for x in self.data]
            for (i, j), p, c in zip(result_indexes, preds, confs):
                self.results.preds[i][j].append(p)
                self.results.confs[i][j].append(c)
        else:
            self.results.preds = [None for _ in self.data]
            self.results.confs = [None for _ in self.data]
            for i, p, c in zip(result_indexes, preds, confs):
                self.results.preds[i] = p
                self.results.confs[i] = c

    def to_raw_examples(
        self, file_format=None, format_fn=None, n=None,
        seed=None, new_sample=True
    ):
        """Flattens all test examples into a single list

        Parameters
        ----------
        file_format : string, must be one of 'jsonl', 'tsv', or None
            None just calls str(x) for each example in self.data
        format_fn : function or None
            If not None, call this function to format each example in self.data
        n : int
            If not None, number of samples to draw
        seed : int
            Seed to use if n is not None
        new_sample: bool
            If False, will rely on a previous sample and ignore the 'n' and
            'seed' parameters

        Returns
        -------
        list(string)
            List of all examples. Indices of example to test case will be
            stored in self.result_indexes. If n is not None, self.run_idxs will
            store the test case indexes.
        """
        if file_format == 'jsonl':
            import json
            format_fn = lambda x, m: json.dumps(x)
        elif file_format == 'tsv':
            format_fn = lambda x, m: '\t'.join(x).replace('\n', ' ')
        else:
            if format_fn is None:
                format_fn = lambda x, m: str(x).replace('\n', ' ')
        if new_sample:
            examples, meta, indices =\
                self.example_list_and_indices(n, seed=seed)
        else:
            raise Exception('Only new samples are supported.')
            # examples, indices = self.recover_example_list_and_indices()
        examples = [format_fn(x, m) for x, m in zip(examples, meta)]
        return examples

    def to_raw_file(
        self, path, task, file_format=None, format_fn=str,
        header=None, n=None, seed=None
    ):
        """Flatten test cases into individual examples and print them to file.
        Indices of example to test case will be stored in self.result_indexes.
        If n is not None, self.run_idxs will store the test case indexes.

        Parameters
        ----------
        path : string
            File path
        file_format : string, must be one of 'jsonl', 'tsv', or None
            None just calls str(x) for each example in self.data
        format_fn : function or None
            If not None, call this function to format each example in self.data
        header : string
            If not None, first line of file
        n : int
            If not None, number of samples to draw
        seed : int
            Seed to use if n is not None
        """
        # file_format can be jsonl, TODO
        # format_fn takes an example and outputs a line(s) in the file
        ret = ''
        if header is not None:
            ret += header.strip('\n') + '\n'

        if task in sequence_tasks:
            if format_fn is not None:
                logger.warning("Replacing given format_fn with a tokenizer.")

            # if the data is pre-tokenized and this is recorded in the
            # metadata, use that tokenization (this ensures the per-token
            # labels match the predictions from the model)
            format_fn =\
                lambda x, m: "\n".join(m[TOKENIZATION_DICT][x]) + "\n" \
                    if TOKENIZATION_DICT in m \
                    else "\n".join(tokenize(x)) + "\n"

        examples = self.to_raw_examples(
            file_format=file_format, format_fn=format_fn, n=n, seed=seed)
        ret += '\n'.join(examples)
        with open(path, 'w') as f:
            f.write(ret)

    def _results_exist(self):
        return hasattr(self, 'results') and self.results

    def _check_results(self):
        if not self._results_exist():
            raise(Exception('No results. Run run() first'))

    def _check_create_results(self, overwrite, check_only=False):
        if self._results_exist() and not overwrite:
            raise(Exception('Results exist. To overwrite, set overwrite=True'))
        if not check_only:
            self.results = Munch()

    def _filter_the_data(self, data_filter: Dict):
        # TODO: this works only if the data wasn't sampled (sampling is not
        # supported yet)
        indx_to_keep = []
        orig_labels = self.labels_dict[self.results.task].labels

        for i in range(len(self.data)):
            m = self.meta[i]
            if all([k in m and m[k] == v for k, v in data_filter.items()]):
                indx_to_keep.append(i)
        if indx_to_keep:
            data, meta, labels, preds, confs = [], [], [], [], []
            for i in indx_to_keep:
                data.append(self.data[i])
                meta.append(self.meta[i])
                labels.append(orig_labels[i])
                preds.append(self.results.preds[i])
                confs.append(self.results.confs[i])
            return data, meta, labels, preds, confs
        return None, None, None, None, None

    def fill_missing_attributes(self):
        """
        Fill required fields which have not been used for present versions
        of the eval core.
        """
        # Older versions of the cores, which don't have data_structure field,
        # all have ungrouped data
        if not hasattr(self, 'data_structure'):
            self.data_structure = DataShape.UNGROUPED
            self.group_names = None
        if not hasattr(self, 'core_record_cache'):
            self.core_record_cache = {}

    def _get_core_record(self, test: AbstractTest):
        """
        Checks the data_filter in the self.results and created a CoreRecord
        with all the data, meta etc. which holds the original data filtered
        according to data_filter. If data_filter is None then the CoreRecord
        holds original data, meta etc.
        """
        data_filter: Dict = self.results.data_filter
        task: str = self.results.task
        data_nclasses: int = self.labels_dict[task].n_classes
        labels: List = self.labels_dict[task].labels

        self.fill_missing_attributes()

        if data_filter and self.meta and \
                len(self.meta) == len(self.data) == len(labels):
            data, meta, labels, preds, confs =\
                self._filter_the_data(data_filter)
        else:
            data, meta, labels, preds, confs =\
                self.data, self.meta, labels, \
                self.results.preds, self.results.confs

        if not data:
            return None

        # if there are no labels for NER it means the sentence is not
        # 'suited' for NER; e.g. it doesn't have any named entities of interest
        # hence, we drop all examples without labels in evaluation
        # drop_none_labels = True if task == "NER" else test.drop_none_labels
        drop_none_labels = True
        cache_entry = CacheEntry(
            str(data_filter), test.required_ds,
            test.probability_based, drop_none_labels,
            test.group_flatten_method)
        if cache_entry in self.core_record_cache:
            return deepcopy(self.core_record_cache[cache_entry])
        else:
            # TODO: fix the run_idx to match the new filtered data
            # -- to support sampling. FOR NOW SAMPLING IS NOT SUPPORTED.
            new_record = CoreRecord(
                # the state has copies for safety (no tests can alter what's
                # in the evaluation core)
                deepcopy(data),
                deepcopy(meta),
                deepcopy(labels),
                deepcopy(preds),
                deepcopy(confs),
                deepcopy(self.label_vocab),
                self.data_structure,
                self.group_names,
                task,
                data_nclasses,
                deepcopy(self.run_idxs)
            )
            # this is done here in order to do it once and store the result
            # in a cache for efficiency
            new_record.process_labels_preds_and_confs(
                test.required_ds, test.probability_based,
                drop_none_labels, test.group_flatten_method)
            self.core_record_cache[cache_entry] = new_record
            return deepcopy(new_record)

    def run_from_preds_confs(
            self, preds, confs, label_vocab=None, task=None,
            data_filter=None, overwrite=False):
        """Update self.results (run tests) from list of predictions and
        confidences

        Parameters
        ----------
        preds : list
            predictions
        confs : list
            confidences
        overwrite : bool
            If False, raise exception if results already exist

        data_filter: a dictionary, e.g. {"DOMAIN": business} -- if this eval
        core has metadata that marks the properties in the dictionary then the
        evaluation will only focus on examples that match the constraints
        """
        if not task:
            raise Exception('Task has to be provided to determine the labels!')
        elif task not in self.labels_dict:
            logger.warning(
                f"Task {task} is lacking labels in this " +
                "evaluation core. This will limit the metrics that " +
                "can be used.")
            if self.data_structure == DataShape.GROUPED:
                labels = [[None] * len(x) for x in self.data]
            else:
                labels = [None] * len(self.data)
            self.labels_dict[task] =\
                Munch({"labels": labels, "n_classes": None})

        self.core_record_cache = {}
        self._check_create_results(overwrite)

        # store results in self.results and label vocab in self.label_vocab
        self.update_results_from_preds(preds, confs)

        self.label_vocab = label_vocab
        self._check_results()

        # the data_filter is saved in results, so that it can retrieved/
        # recognized in summary() function
        self.results.data_filter = data_filter
        self.results.task = task

        for test in self.tests:
            try:
                # get the record for the test -- keeping this in the loop is
                # less efficient, but more safe -- each test gets it's own
                # copy of a record (with accordingly processed data)
                core_record = self._get_core_record(test)

                if (core_record.sequence and
                        issubclass(type(test), BasicClassificationMetrics)) or \
                        (not core_record.sequence and
                            issubclass(type(test), BasicSeqMetrics)):
                    continue

                logger.info(f"Evaluating on {test.name}...")
                # this updates the self.results field with the new results
                test_res = test.compute(core_record)
                self.results[test.get_name()] = test_res
            except Exception as err:
                logger.error(f"Couldn't run the {test.name} test.")
                # all 'non-expected'/'non-standard' Exceptions are not types
                # of the catch-all Exception class
                if type(err) != Exception:
                    logger.exception(err)
                else:
                    logger.warning(err)
                self.results[test.get_name()] = err

    def run_from_file(
        self, path, file_format=None, format_fn=None, task=None,
        ignore_header=False, overwrite=False, data_filter=None
    ):
        """Update self.results (run tests) from a prediction file

        Parameters
        ----------
        path : string
            prediction file path
        file_format : string
            None, or one of 'pred_only', 'softmax', binary_conf',
                'pred_and_conf', 'pred_and_softmax', 'squad',
                pred_only: each line has a prediction
            softmax: each line has prediction probabilities separated by spaces
            binary_conf: each line has the prediction probability of class 1
                (binary)
            pred_and_conf: each line has a prediction and a confidence value,
                separated by a space
            pred_and_softmax: each line has a prediction and all softmax
                probabilities, separated by a space
            squad: TODO
        format_fn : function
            If not None, function that reads a line in the input file and
                outputs a tuple of (prediction, confidence)
        ignore_header : bool
            If True, skip first line in the file
        overwrite : bool
            If False, raise exception if results already exist
        """
        # file_format can be 'pred_only' (only preds, conf=1), TODO
        # Format_fn takes a line in the file and outputs (pred, conf)
        # Checking just to avoid reading the file in vain
        self._check_create_results(overwrite, check_only=True)

        pred_file_out = read_pred_file(
            path, file_format=file_format,
            format_fn=format_fn,
            ignore_header=ignore_header)

        if len(pred_file_out) == 2:
            preds, confs = pred_file_out
            label_vocab = None
        elif len(pred_file_out) == 3:
            preds, confs, label_vocab = pred_file_out
        else:
            raise Exception('Incorrect output of the read_pred_file function.')

        self.run_from_preds_confs(
            preds, confs, label_vocab, task=task,
            data_filter=data_filter, overwrite=overwrite)

    # TODO: fix. Only running from a file is supported atm.
    # def run(
    #     self, predict_and_confidence_fn, overwrite=False, verbose=True,
    #     n=None, seed=None
    # ):
    #     """Runs test
    #
    #     Parameters
    #     ----------
    #     predict_and_confidence_fn : function
    #         Takes as input a list of examples
    #         Outputs a tuple (predictions, confidences)
    #     overwrite : bool
    #         If False, raise exception if results already exist
    #     verbose : bool
    #         If True, print extra information
    #     n : int
    #         If not None, number of samples to draw
    #     seed : int
    #         Seed to use if n is not None
    #     """
    #     # Checking just to avoid predicting in vain, will be created in
    #     # run_from_preds_confs
    #     self._check_create_results(overwrite, check_only=True)
    #     examples, result_indexes = self.example_list_and_indices(n, seed=seed)
    #
    #     if verbose:
    #         print('Predicting %d examples' % len(examples))
    #     preds, confs = predict_and_confidence_fn(examples)
    #     self.run_from_preds_confs(preds, confs, overwrite=overwrite)

    def summary(self, **kwargs):
        for i, test in enumerate(self.tests):
            print(f"\n===================\nTEST {i+1}:", test.name)
            try:
                tname = test.get_name()
                if tname not in self.results:
                    logger.warning(f"Missing results for test {tname}")
                else:
                    res_munch = self.results[tname]

                    # missing res
                    if type(res_munch) == Exception:
                        logger.warning(
                            f"Missing results for test {tname}.")
                        logger.exception(f"Reason: {res_munch}")
                        continue

                    core_record = self._get_core_record(test)

                    test.summary(core_record, res_munch, **kwargs)
            except Exception as err:
                logger.warning(f"Missing results for the test. Reason: {err}")
