from typing import List, Union, Dict
from .utils import sequence_tasks, DataShape, tokenize, \
    get_class_from_seq_label, FlattenGroup, is_1d_list, MAX_N_GROUPINGS, \
    is_2d_list
import numpy as np
from copy import deepcopy
from collections import defaultdict, namedtuple
from itertools import product
from multiprocessing import Pool

import regex as re

import logging
logger = logging.getLogger(__name__)


class CoreRecord():
    """
    A class which instances are passed to the tests -- it holds all fields
    that are relevant when a metric/test are computed. Importantly, each test
    gets a separate instance of this class, so they can change, restructure
    the data etc.
    """
    def __init__(
        self, data, meta, labels,
        preds, confs,
        label_vocab,
        data_structure,
        group_names,
        task,
        n_classes,
        run_idxs
    ) -> None:
        self.data = data
        self.meta = meta
        self.labels = labels

        self.data_nclasses = n_classes
        self.preds = preds
        self.confs = confs

        self.task = task
        self.run_idxs = run_idxs

        sequence = task in sequence_tasks
        self.sequence = sequence

        # for text classification labels are ints, but for sequence labeling
        # they are (lists of) strings -- this is to easily recognise different
        # labels for the same entity types (e.g. B-PER, I-PER, L-PER)
        if not label_vocab:
            if sequence:
                self.label_vocab = [i for i in range(len(confs[0][0][0]))]
            else:
                self.label_vocab = [i for i in range(len(confs[0][0]))]
        else:
            if all([type(x) == str and x.isdigit() for x in label_vocab]):
                label_vocab = [int(x) for x in label_vocab]
            self.label_vocab = label_vocab

        # at all times, keep track of the structure of the data/preds/confs
        # GROUPED: #ngorups x #nexamples vs UNGROUPED: #nexamples x #ngroups
        self.data_structure = data_structure

        if group_names:
            self.group_names = group_names
        else:
            # the data is not grouped (if it's grouped group_names are pre-set)
            # the group names might be retrieved from the meta data
            n_groups = len(self.preds[0])
            if self.meta:
                self.group_names = [self.meta[0][x] for x in range(n_groups)]
            else:
                self.group_names = [f"Group{x+1}" for x in range(n_groups)]

    def label_meta(self, i):
        if self.data_structure == DataShape.GROUPED:
            raise Exception('Function label_meta is unsupported for non ' +
                            'matched data (CoreRecord).')

        if self.labels is None:
            label = None
        else:
            label = self.labels if type(self.labels) not in [list, np.array] \
                else self.labels[i]
        if self.meta is None:
            meta = None
        else:
            meta = self.meta if type(self.meta) not in [list, np.array] \
                else self.meta[i]
        return label, meta

    def get_data_for_many_groups(self, gindex_to_skip=None):
        new_labels, new_preds, new_confs, new_meta = [], [], [], []

        for gindex, (labels, preds, confs, meta) in enumerate(zip(
                self.labels, self.preds, self.confs, self.meta)):
            if gindex_to_skip is not None and gindex == gindex_to_skip:
                continue

            new_labels += [x for x in labels]
            new_preds += [x for x in preds]
            new_confs += [x for x in confs]
            new_meta += [x for x in meta]
        return new_labels, new_preds, new_confs, new_meta

    def get_n_examples(self) -> Union[int, Dict[str, int]]:
        if self.data_structure == DataShape.GROUPED:
            if all([x == self.labels[0] for x in self.labels]):
                # all groups have the same number of examples
                return len(self.labels[0])
            else:
                # the data must have been originally 'grouped' to allow for a
                # mismatch OR there are differing number of terms for different
                # groups
                return {k: len(self.labels[i])
                        for i, k in enumerate(self.group_names)}
        elif self.data_structure == DataShape.UNGROUPED:
            return len(self.labels)
        else:
            raise Exception(
                f'Unsupported data structure: {self.data_structure}')

    def get_classes(self):
        """
        Return the classes. Note that classes are different from labels; e.g.
        for sequence labeling this function merges differnt labels for a single
        class, e.g. B-PER, I-PER etc. into PER.
        """
        if self.sequence:
            classes = {get_class_from_seq_label(lab)
                       for lab in self.label_vocab}
            return sorted(classes)
        else:
            return self.label_vocab

    def process_labels_preds_and_confs(
            self, required_ds: DataShape, probability_based: bool,
            drop_none_labels: bool, group_flatten_method=None) -> None:
        """
        (Potentially) restructure the labels, preds and confs stored in the
        core_record. Note: each test gets its own CoreRecord instance, so other
        tests won't be affected by altering fields in core_record.

        Return the number of remaining test examples overall (int) or per
        group (dict).
        """
        if required_ds == DataShape.GROUPED:
            if self.data_structure != DataShape.GROUPED:
                # #examples x #groups => #groups x #examples
                self.group_data()
        elif required_ds == DataShape.UNGROUPED and \
                self.data_structure == DataShape.GROUPED:
            raise Exception(
                'Cannot run a test for ungrouped/counterfatual data on ' +
                'data split into groups!')
        elif required_ds != DataShape.UNGROUPED:
            raise Exception(f'Unsupported data structure: {required_ds}')

        # this only applies when the structure is
        # ngroups x ntemplate_examples x nvariations or
        # ntemplate_examples x ngroups x nvariations
        # if there is only one variation this doesn't do anything
        self.flatten_within_group(group_flatten_method)

        # if labels will be used in a metric (i.e. none labels are dropped)
        # the labels' classes must match the model's classes. Here, they
        # are adjusted if such adjustment is supported for a given task
        if self.task in ["SENT"]:
            n_classes = len(self.get_classes())
            self.adjust_for_class_eval(n_classes)

        if drop_none_labels:
            self.filter_class_out(class_to_filter=None)
            if all([x == [] for x in self.labels]):
                raise Exception('No data suitable for this evaluation.')

        if self.sequence and probability_based:
            # TODO: this is NER specific - the code should be restructured
            # when more tasks are added

            # accumulate the probabilities for each class
            # e.g. sum the probabilities for B-ORG, I-ORG, L-ORG
            self.accumulate_ner_probs()
            self.average_conf_across_tokens_for_identity_term()

    #################################
    #      FILTERING THE DATA/      #
    #     ADJUSTING THE VALUES      #
    #################################

    def filter_class_out(self, class_to_filter):
        """
        Filter the cases for which the labels is 'class_to_filter'.
        """
        def helper(xs, indices_to_keep):
            xs = [xs[i] for i in indices_to_keep]
            return xs

        if self.data_structure == DataShape.GROUPED:
            for gidx in range(len(self.labels)):
                labels = np.array(self.labels[gidx], dtype=object)
                indices_to_keep = np.argwhere(labels != None)
                indices_to_keep = [x[0] for x in indices_to_keep]
                self.labels[gidx] = list(labels[indices_to_keep])
                self.data[gidx] = helper(self.data[gidx], indices_to_keep)
                self.preds[gidx] = helper(self.preds[gidx], indices_to_keep)
                self.confs[gidx] = helper(self.confs[gidx], indices_to_keep)
                self.meta[gidx] = helper(self.meta[gidx], indices_to_keep)
        else:
            labels = np.array(self.labels, dtype=object)
            indices_to_keep = np.argwhere(labels != None)
            indices_to_keep = [x[0] for x in indices_to_keep]
            self.data = helper(self.data, indices_to_keep)
            self.labels = list(labels[indices_to_keep])
            self.preds = helper(self.preds, indices_to_keep)
            self.confs = helper(self.confs, indices_to_keep)
            self.meta = helper(self.meta, indices_to_keep)

    def adjust_for_class_eval(
        self,
        n_classes: int,
        filter_none=True
    ) -> None:
        """
        Adjusts the labels, preds and confs in the core_record based on the
        number of classes.

        If n_classes is 2, all neutral examples (class 1) are removed
        (if filter_none=True) or their class is set to None
        (if filter_none=True). All positive examples (class 2) are turned to
        class 1.

        If n_classes is 5 or 7 then the predictions of the model are adjusted
        to cover only 3 classes: positive, neutral, negative.

        If filter_none=True then all examples without a specified class are
        removed.

        IMPORTANT: The support is only for templates that have 3 classes.
        """

        data_nclasses = self.data_nclasses
        all_none = lambda x: all(v is None for v in x)

        def turn_3_class_labels_to_2_class(tmp_labels):
            new_labels = []
            for x in tmp_labels:
                if x == 0:
                    new_labels.append(x)
                elif x == 1 or x is None:
                    # none will be filtered out if necessary
                    new_labels.append(None)
                elif x == 2:
                    new_labels.append(1)
            return new_labels

        def turn_2_class_labels_to_3_class(tmp_labels):
            new_labels = []
            for x in tmp_labels:
                if x == 0:
                    new_labels.append(x)
                elif x == 1:
                    new_labels.append(2)
            return new_labels

        def reduce_preds_to_3_class(tmp_preds):
            new_pred = []
            mid = int(n_classes/2)
            for x in tmp_preds:
                if x == mid:
                    new_pred.append(1)
                elif x < mid:
                    new_pred.append(0)
                else:
                    new_pred.append(2)
            return new_pred

        if self.data_nclasses == n_classes:
            pass

        elif self.data_structure == DataShape.GROUPED:
            n_groups = len(self.labels)
            if self.data_nclasses is None and all(
                    all_none(self.labels[gidx]) for gidx in range(n_groups)):
                pass
            elif n_classes == 2 and data_nclasses == 3:
                for gidx in range(n_groups):
                    self.labels[gidx] =\
                        turn_3_class_labels_to_2_class(self.labels[gidx])
            elif n_classes == 3 and data_nclasses == 2:
                for gidx in range(len(self.labels)):
                    self.labels[gidx] =\
                        turn_2_class_labels_to_3_class(self.labels[gidx])
            elif n_classes in [5, 7] and data_nclasses == 3:
                for gidx in range(len(self.labels)):
                    self.preds[gidx] =\
                        reduce_preds_to_3_class(self.preds[gidx])
            else:
                raise Exception(
                    f'Unsupported evalution for data #classes: \
                    {data_nclasses} and model #classes: {n_classes}')
        else:
            if self.data_nclasses is None and all_none(self.labels):
                pass
            elif n_classes == 2 and data_nclasses == 3:
                self.labels = turn_3_class_labels_to_2_class(self.labels)
            elif n_classes == 3 and data_nclasses == 2:
                self.labels = turn_2_class_labels_to_3_class(self.labels)
            elif n_classes in [5, 7] and data_nclasses == 3:
                for eidx in range(len(self.labels)):
                    self.preds[eidx] =\
                        reduce_preds_to_3_class(self.preds[eidx])
            else:
                raise Exception(
                    f'Unsupported evalution for data #classes: \
                    {data_nclasses} and model #classes: {n_classes}')

    #####################################
    #      RESTRUCTURING THE DATA       #
    #####################################

    def flatten_within_group(self, flatten_group):
        if flatten_group in [None, FlattenGroup.NONE] or \
                is_1d_list(self.data[0]):
            return

        new_preds = []
        new_confs = []
        new_data = []
        new_meta = []
        new_labels = []

        if flatten_group == FlattenGroup.RANDOM_MATCH:
            assert self.data_structure == DataShape.UNGROUPED

            thresh = MAX_N_GROUPINGS
            for eidx in range(len(self.data)):
                # keeping the sampling in the inner loop allows for number
                # of samples for each group to differ from template to template
                # but also causes the samples to differ from temp. to temp.
                sample_idxs = [list(range(len(x))) for x in self.data[eidx]]

                # detect whether there is too many options while avoiding
                # overflow
                total_options = 1
                for x in sample_idxs:
                    total_options *= len(x)
                    if total_options > thresh:
                        break

                if total_options > thresh:
                    groupings = set()
                    while len(groupings) < thresh:
                        sample = tuple(
                            [np.random.choice(x) for x in sample_idxs])
                        groupings.add(sample)
                else:
                    groupings = set(product(*sample_idxs))

                meta = self.meta[eidx]
                for chosen_idxs in groupings:
                    new_data.append(
                        [self.data[eidx][gidx][cidx]
                         for gidx, cidx in enumerate(chosen_idxs)])

                    new_preds.append(
                        [self.preds[eidx][gidx][cidx]
                         for gidx, cidx in enumerate(chosen_idxs)])

                    new_confs.append(
                        [self.confs[eidx][gidx][cidx]
                         for gidx, cidx in enumerate(chosen_idxs)])

                    tmp_meta = deepcopy(meta)
                    for gidx, cidx in enumerate(chosen_idxs):
                        gname = meta[gidx]
                        tmp_meta["SAMPLE"][gname] = meta["SAMPLE"][gname][cidx]
                    new_meta.append(tmp_meta)
                    new_labels.append(self.labels[eidx])

        elif flatten_group == FlattenGroup.FLATTEN:
            assert self.data_structure == DataShape.GROUPED
            for gidx in range(len(self.group_names)):
                new_group_preds, new_group_confs, new_group_data, \
                    new_group_labels, new_group_meta = [], [], [], [], []
                for eidx in range(len(self.labels[gidx])):
                    nversions = len(self.preds[gidx][eidx])
                    new_group_preds += list(self.preds[gidx][eidx])
                    new_group_confs += list(self.confs[gidx][eidx])
                    new_group_data += list(self.data[gidx][eidx])
                    new_group_labels += [self.labels[gidx][eidx]] * nversions

                    for nv in range(nversions):
                        tmp_meta = deepcopy(self.meta[gidx][eidx])
                        gname = tmp_meta[gidx]
                        tmp_meta["SAMPLE"] =\
                            {gname: tmp_meta["SAMPLE"][gname][nv]}
                        tmp_meta["GROUP_FILL"] = tmp_meta["GROUP_FILL"][nv]
                        new_group_meta.append(tmp_meta)
                new_preds.append(new_group_preds)
                new_confs.append(new_group_confs)
                new_labels.append(new_group_labels)
                new_meta.append(new_group_meta)
                new_data.append(new_group_data)
        elif flatten_group == FlattenGroup.AVERAGE:
            assert self.data_structure == DataShape.UNGROUPED

            n_groups = len(self.data[0])
            new_labels = self.labels
            new_meta = self.meta

            for eidx in range(len(self.data)):
                meta = self.meta[eidx]
                new_data.append([self.meta[eidx]["TEMPLATE"]] * n_groups)

                new_preds.append(
                    [np.mean(np.array(self.preds[eidx][gidx]), axis=0)
                     for gidx in range(n_groups)])
                new_confs.append(
                    [np.mean(np.array(self.confs[eidx][gidx]), axis=0)
                     for gidx in range(n_groups)])
        elif flatten_group == FlattenGroup.FLATTEN_ALL:
            # this loses all structure; i.e. group divisions and example
            # divisions
            for gidx in range(len(self.data)):
                for eidx in range(len(self.data[gidx])):
                    for vidx in range(len(self.data[gidx][eidx])):
                        new_preds.append([self.preds[gidx][eidx][vidx]])
                        new_confs.append([self.confs[gidx][eidx][vidx]])
                        new_data.append([self.data[gidx][eidx][vidx]])

                        if self.data_structure == DataShape.GROUPED:
                            new_labels.append(self.labels[gidx][eidx])
                            tmp_meta = deepcopy(self.meta[gidx][eidx])
                            gname = tmp_meta[gidx]
                            tmp_meta["SAMPLE"] =\
                                {gname: tmp_meta["SAMPLE"][gname][vidx]}
                            tmp_meta["GROUP_FILL"] = tmp_meta["GROUP_FILL"][vidx]
                            new_meta.append(tmp_meta) 
                        else:
                            new_labels.append(self.labels[gidx])
                            new_meta.append(self.meta[gidx])

        self.labels = new_labels
        self.preds = new_preds
        self.confs = new_confs
        self.meta = new_meta
        self.data = new_data

    def _group(self, data) -> List:
        """
        Regroups the data in a format: #tempaltes sents x #groups
        to: #groups x #templates sents
        """
        if not data:
            return data

        if self.data_structure != DataShape.GROUPED:
            n_groups = len(self.group_names)
            assert len(data[0]) == n_groups

            if not all([type(x) in [np.ndarray, list] and len(x) == n_groups
                        for x in data]):
                raise Exception(
                    'Uneven number of groups across sentences.')

            return [list(t) for t in zip(*data)]
        else:
            return data

    def group_data(self):
        """
        Regroups labels, preds and confs data in a format:
        #tempaltes sents x #groups to: #groups x #templates sents
        """
        if self.data_structure == DataShape.GROUPED:
            return

        n_groups = len(self.group_names)
        self.confs = self._group(self.confs)
        self.preds = self._group(self.preds)
        self.data = self._group(self.data)
        assert len(self.preds) == len(self.confs) == len(self.data) == n_groups

        self.labels = [self.labels] * n_groups

        # NOTE: this adjustment of metadata works for metadata from templates
        if self.meta:
            new_meta = []
            for i in range(n_groups):
                new_meta_for_group = []

                for m in self.meta:
                    new_m = deepcopy(m)
                    gname = m[i]
                    if "." in gname:
                        prop, term = gname.split(".")
                        gfill = m['SAMPLE'][prop][term]
                    else:
                        gfill = m['SAMPLE'][gname]
                    # fill in the fill slot for the group
                    new_m['GROUP_FILL'] = gfill
                    new_meta_for_group.append(new_m)
                new_meta.append(new_meta_for_group)
            self.meta = new_meta

        self.data_structure = DataShape.GROUPED

    def _simplify_seq_labels(self):
        if not self.sequence:
            raise Exception('The task is not a sequence labeling taks. \
                Cannot simplify labels.')

        if self.data_structure == DataShape.GROUPED:
            for gidx in range(len(self.labels)):
                labels = self.labels[gidx]
                for eidx, ex_labels in enumerate(labels):
                    if ex_labels is None:
                        continue
                    for tidx, t_label in enumerate(ex_labels):
                        self.labels[gidx][eidx][tidx] =\
                            get_class_from_seq_label(t_label)
        else:
            for eidx, ex_labels in enumerate(self.labels):
                if ex_labels is None:
                    continue
                for tidx, t_label in enumerate(ex_labels):
                    self.labels[eidx][tidx] =\
                        get_class_from_seq_label(t_label)

    def accumulate_ner_probs(self):
        """
        Accumulate the probabilities for each class
        e.g. sum the probabilities for B-ORG, I-ORG, L-ORG.
        It alters the fields in the core_record.

        IMPORTANT: this also changes predictions and labels (makes them less
        fine-grained, e.g. ORG instead of B-ORG)
        """
        new_softmax_vocab = self.get_classes()
        label_vocab = self.label_vocab

        # changing confs can change predicted classes (e.g. entity types)
        # althouth it's quite unlikely
        changed_pred_class = 0
        all_tokens = 0

        def acc_probs_for_token(token_confs, token_pred):
            tag2prob = defaultdict(int)
            for p, prob in enumerate(token_confs):
                # skip the type of label B-, I- etc.
                label = get_class_from_seq_label(label_vocab[p])
                tag2prob[label] += prob
            new_confs = np.array([tag2prob[x] for x in new_softmax_vocab])
            new_pred_idx = np.argmax(new_confs)
            new_pred = new_softmax_vocab[new_pred_idx]
            return new_confs, new_pred

        # this loop works for both grouped and ungrouped data
        for e, example_confs in enumerate(self.confs):
            if type(self.confs[e]) == np.ndarray:
                self.confs[e] = self.confs[e].tolist()

            for g, group_example_confs in enumerate(example_confs):
                if type(self.confs[e][g]) == np.ndarray:
                    self.confs[e][g] = self.confs[e][g].tolist()

                if is_1d_list(group_example_confs[0]):
                    for t, token_confs in enumerate(group_example_confs):
                        new_confs, new_pred = acc_probs_for_token(
                            token_confs, self.preds[e][g][t])
                        current_pred_class =\
                            get_class_from_seq_label(self.preds[e][g][t])
                        self.confs[e][g][t] = new_confs
                        self.preds[e][g][t] = new_pred

                        if new_pred != current_pred_class:
                            changed_pred_class += 1
                        all_tokens += 1
                else:
                    # for each group there are many versions of each example
                    for v, group_example_version_confs in enumerate(
                            group_example_confs):
                        if type(self.confs[e][g][v]) == np.ndarray:
                            self.confs[e][g][v] = self.confs[e][g][v].tolist()

                        for t, token_confs in enumerate(
                                group_example_version_confs):
                            new_confs, new_pred = acc_probs_for_token(
                                token_confs, self.preds[e][g][v][t])
                            current_pred_class = get_class_from_seq_label(
                                self.preds[e][g][v][t])
                            self.confs[e][g][v][t] = new_confs
                            self.preds[e][g][v][t] = new_pred

                            if new_pred != current_pred_class:
                                changed_pred_class += 1
                            all_tokens += 1

        if changed_pred_class > 0:
            logger.warning(f'Changed prediction for \
                {changed_pred_class}/{all_tokens} tokens after merging confs \
                for each entity type.')

        self.label_vocab = new_softmax_vocab
        self._simplify_seq_labels()

    def _merge_tokens_for_identity_term(
            self, sent_confs, sent_preds, sent_meta, gidx, vidx=None):
        ikey = sent_meta['IDENTITY_KEY']

        if "TOKENIZED_TEMPLATE" in sent_meta:
            temp_toks = sent_meta["TOKENIZED_TEMPLATE"]
        else:
            temp_toks = tokenize(sent_meta["TEMPLATE"])

        gname = sent_meta[gidx]
        if "." in gname:
            prop, term = gname.split(".")
            if vidx is None:
                gfill = str(sent_meta['SAMPLE'][prop][term])
            else:
                gfill = str(sent_meta['SAMPLE'][prop][vidx][term])
        else:
            if vidx is None:
                gfill = str(sent_meta['SAMPLE'][gname])
            else:
                gfill = str(sent_meta['SAMPLE'][gname][vidx])

        # TODO: there can be situations where the tokenization is context-
        # dependent; for now we don't handle such cases
        fill_toks = tokenize(gfill)

        new_confs = []
        new_preds = []
        p = 0
        for temp_tok in temp_toks:
            if temp_tok == f"@{ikey}@":
                newc = []
                for x in range(p, p + len(fill_toks)):
                    # make sure it's an np.ndarray, to sum correctly
                    newc.append(np.array(sent_confs[x]))

                new_conf = sum(newc)/len(newc)
                new_confs.append(new_conf)

                new_pred_idx = np.argmax(new_conf)
                new_pred = self.label_vocab[new_pred_idx]
                new_preds.append(new_pred)
                p += len(fill_toks)
            else:
                new_confs.append(sent_confs[p])
                new_preds.append(sent_preds[p])
                p += 1
        return new_confs, new_preds

    def average_conf_across_tokens_for_identity_term(self):
        """
        Average the scores for all multi-word identity tokens -> turn them into
        one score for that identity. The shapes of the resulting confs for each
        group should be the same as the labels.

        This can only be used on sequence data.

        WARNING: this function should only be called after \
            accumulate_ner_probs is called (!). It adapts the prediction for
            the new identity tokens and if the B- I- L- etc. labels are still
            in the softmax, the will lead to errors.

        """
        assert not any(["-" in cl for cl in self.label_vocab])
        data_structure = self.data_structure

        new_confs = []
        new_preds = []
        for gidx, (group_confs, group_preds, group_meta) in \
                enumerate(zip(self.confs, self.preds, self.meta)):
            new_group_confs = []
            new_group_preds = []

            if data_structure == DataShape.UNGROUPED:
                # group meta is actually sent_meta
                group_meta = [group_meta] * len(group_confs)

            for eidx, (sent_confs, sent_preds, sent_meta) in \
                    enumerate(zip(group_confs, group_preds, group_meta)):

                if data_structure == DataShape.UNGROUPED:
                    gidx = eidx

                if is_1d_list(sent_confs[0]):
                    new_sent_confs, new_sent_preds =\
                        self._merge_tokens_for_identity_term(
                            sent_confs, sent_preds, sent_meta, gidx)
                else:
                    new_sent_confs = []
                    new_sent_preds = []
                    for vidx, (sent_v_confs, sent_v_preds) in \
                            enumerate(zip(sent_confs, sent_preds)):
                        new_sent_v_confs, new_sent_v_preds =\
                            self._merge_tokens_for_identity_term(
                                sent_v_confs, sent_v_preds, sent_meta,
                                gidx, vidx)
                        new_sent_confs.append(new_sent_v_confs)
                        new_sent_preds.append(new_sent_v_preds)
                new_group_confs.append(new_sent_confs)
                new_group_preds.append(new_sent_preds)

            new_confs.append(new_group_confs)
            new_preds.append(new_group_preds)

        self.confs = new_confs
        self.preds = new_preds
