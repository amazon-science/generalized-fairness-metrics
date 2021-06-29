from munch import Munch

import checklist_fork.checklist.tests.sig_functions as sfuns
from functools import partial
from ..utils import group_data, accumulate_ner_probs, \
    DataShape, average_conf_across_tokens_for_identity_term, \
    flatten_confs_and_labels

from .abstract_test import AbstractTest
from checklist_fork.checklist.eval_core import CoreRecord


###########################
#       BASE CLASS        #
###########################

class SignificanceTest(AbstractTest):
    def __init__(
        self, sig_fun, sig_threshold=0.05, n_groups=2, name=None, cl=None
    ) -> None:
        self.sig_fun = sig_fun
        self.sig_threshold = sig_threshold
        self.n_groups = n_groups
        self.name = name

        # run the test only on examples which have this class as gold class
        # and use the probability for that class as a score for tests
        self.cl = cl

    def compute(self, core_record: CoreRecord) -> Munch:
        if core_record.data_structure != DataShape.UNGROUPED:
            raise Exception('Cannot run a test for ungrouped/counterfatual ' +
                            'data on data split into groups!')

        group_data(core_record)
        if core_record.sequence:
            # TODO: check if accumulate works ok with regrouped data
            accumulate_ner_probs(core_record)

            # get rid of token mismatches
            average_conf_across_tokens_for_identity_term(core_record)

            confs, labels = flatten_confs_and_labels(
                core_record.confs, core_record.labels,
                core_record.data_structure)
        else:
            confs = core_record.confs
            labels = core_record.labels

        if self.cl:
            idx = core_record.label_vocab.index(self.cl)
            new_confs = []

            for group_confs, group_labels in zip(confs, labels):
                group_confs = [x for i, x in enumerate(group_confs)
                               if group_labels[i] is not None]

                process_label =\
                    lambda x: x if not core_record.sequence or "-" not in x \
                    else x[2:]

                # add x[idx] twice because sig functions take 1st element
                # TODO: this is unelegant (need to fix)
                group_confs =\
                    [[x[idx], x[idx]] for i, x in enumerate(group_confs)
                     if process_label(group_labels[i]) == self.cl]
                new_confs.append(group_confs)
            confs = new_confs

        stat, pval, info = self.sig_fun(confs)

        # fail when the difference *is* statistically significant
        if pval <= self.sig_threshold:
            passed = False
        else:
            passed = True

        return Munch({"results": (stat, pval, info), "passed": passed})

    def summary(self, core_record, res_munch, **kwargs):
        passed = res_munch.passed
        stat, pval, info = res_munch.results

        pass_str = "PASSED" if passed else "FAILED"
        print(f'Threshold: {self.sig_threshold}')
        print(f'Significance test {self.name}: {pass_str}')
        print(f'Statistic: {stat:.4f}, p-value: {pval:.4f}')
        if info:
            print(f'\nInfo: {info}')

    def save_summary(self, core_record, **kwargs) -> None:
        pass


###########################
#      SPECIFIC TESTS     #
###########################

class PairedTtest(SignificanceTest):
    def __init__(self, sig_threshold=0.05, **kwargs):
        name = "paired t-test"
        super().__init__(
            sfuns.paired_ttest, sig_threshold, 2, name=name, **kwargs)


class Wilcoxon(SignificanceTest):
    def __init__(self, sig_threshold=0.05, **kwargs):
        name = "wilcoxon"
        super().__init__(
            sfuns.wilcoxon, sig_threshold, 2, name=name, **kwargs)


class PairedPermutation(SignificanceTest):
    def __init__(self, sig_threshold=0.05, stat='mean', **kwargs):
        sig_fun = partial(sfuns.permutation_one_sample, stat=stat)
        name = f"paired permutation ({stat})"
        super().__init__(sig_fun, sig_threshold, 2, name=name, **kwargs)


class Permutation(SignificanceTest):
    def __init__(self, sig_threshold=0.05, stat='mean', **kwargs):
        sig_fun = partial(sfuns.permutation_two_sample, stat=stat)
        name = f"permutation ({stat})"
        super().__init__(sig_fun, sig_threshold, 2, name=name, **kwargs)
