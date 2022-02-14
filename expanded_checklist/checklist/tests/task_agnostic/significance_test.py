from munch import Munch

import expanded_checklist.checklist.tests.task_agnostic.sig_functions as sfuns
from functools import partial
from expanded_checklist.checklist.utils import DataShape, \
    flatten_confs_and_labels, get_class_from_seq_label, FlattenGroup

from ..abstract_tests import AbstractTest
from expanded_checklist.checklist.eval_core import CoreRecord

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rc
import math
import numpy as np

###########################
#       BASE CLASS        #
###########################


class SignificanceTest(AbstractTest):
    def __init__(
        self,
        sig_fun,
        dependent_samples: bool,
        sig_threshold=0.05,
        name=None,
        cl=None,
        evaluate_on_all: bool = True
    ) -> None:
        self.sig_fun = sig_fun
        self.sig_threshold = sig_threshold
        self._name = name

        # the data shape depends on the type of test
        if dependent_samples:
            self._required_ds: DataShape = DataShape.UNGROUPED
            self._group_flatten_method = FlattenGroup.AVERAGE
        else:
            # separate the data into separate groups, not organized/tied by the
            # source examples
            self._required_ds: DataShape = DataShape.GROUPED
            self._group_flatten_method = FlattenGroup.FLATTEN

        self._drop_none_labels = False
        self._probability_based = True

        # use the probability for that class as a score for tests
        self.cl = cl
        # run the test only on examples which have this class as gold class
        self.evaluate_on_all = evaluate_on_all

    def compute(self, core_record: CoreRecord) -> Munch:
        if core_record.data_structure != self._required_ds:
            raise Exception('Incorrect data shape for the test.')

        core_record.group_data()
        if core_record.sequence:
            confs, labels = flatten_confs_and_labels(
                core_record.confs, core_record.labels,
                core_record.data_structure)
        else:
            confs = core_record.confs
            labels = core_record.labels

        if self.cl is not None:
            idx = core_record.label_vocab.index(self.cl)

            if not self.evaluate_on_all:
                new_confs = []
                for group_confs, group_labels in zip(confs, labels):
                    process_label =\
                        lambda x: x if x is None or not core_record.sequence \
                        else get_class_from_seq_label(x)
                    group_confs = [
                        x for i, x in enumerate(group_confs)
                        if process_label(group_labels[i]) == self.cl]
                    new_confs.append(group_confs)
                confs = new_confs
        else:
            idx = None

        cl = self.cl if not self.evaluate_on_all else None
        #   self.plot_distributions_from_core_record(idx, core_record, cl)

        # print(core_record.group_names)
        stat, pval, info = self.sig_fun(confs, idx=idx)

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
        print(f'Statistic: {stat:.4f}, p-value: {pval:.4f} ({pval:.2e})')
        if info:
            print(f'\nInfo: {info}')

    def save_summary(self, core_record, **kwargs) -> None:
        pass

    def plot_distributions_from_core_record(
            self, cl_idx, core_record, cl=None):
        # suited for grouped data atm
        n_rows = math.ceil(len(core_record.group_names)/3)
        fig, axs = plt.subplots(
            n_rows, 3, sharex=True, sharey=True, figsize=(12, 4 * n_rows))

        for i, (gname, confs, labels, meta) in enumerate(
                zip(core_record.group_names,
                    core_record.confs, core_record.labels, core_record.meta)):
            # wrongs = [x['TEMPLATE'] for j, x in enumerate(meta)
            #           if labels[j] == 1 and confs[j][1] < 0.5]
            if cl is not None:
                confs = [x[cl_idx] for j, x in enumerate(confs)
                         if labels[j] == cl]
            else:
                confs = [x[cl_idx] for j, x in enumerate(confs)]

            row = i // 3
            col = i % 3

            ax = axs[row, col] if n_rows > 1 else axs[col]
            ax.set_title(gname)
            sns.distplot(
                confs, hist=True, rug=False, kde=False, bins=100, ax=ax)

        plt.tight_layout()
        plt.show()


###########################
#      SPECIFIC TESTS     #
###########################

class PairedTtest(SignificanceTest):
    def __init__(self, sig_threshold=0.05, cl=1, **kwargs):
        name = "paired t-test"
        super().__init__(
            sfuns.paired_ttest, True, sig_threshold,
            name=name, cl=cl, **kwargs)


class Wilcoxon(SignificanceTest):
    def __init__(self, sig_threshold=0.05, cl=1, **kwargs):
        name = "wilcoxon"
        super().__init__(
            sfuns.wilcoxon, True, sig_threshold, name=name, cl=cl, **kwargs)


class PairedPermutation(SignificanceTest):
    def __init__(self, sig_threshold=0.05, stat='mean', cl=1, **kwargs):
        sig_fun = partial(sfuns.permutation_one_sample, stat=stat)
        name = f"paired permutation ({stat})"
        super().__init__(
            sig_fun, True, sig_threshold, name=name, cl=cl, **kwargs)


class Permutation(SignificanceTest):
    def __init__(self, sig_threshold=0.05, stat='mean', cl=1, **kwargs):
        sig_fun = partial(sfuns.permutation_two_sample, stat=stat)
        name = f"permutation ({stat})"
        super().__init__(
            sig_fun, False, sig_threshold, name=name, cl=cl, **kwargs)


class MannWhitneyU(SignificanceTest):
    def __init__(self, sig_threshold=0.05, cl=1, **kwargs):
        name = "mann whitney u"
        super().__init__(
            sfuns.mann_whitney_u, False, sig_threshold, name=name,
            cl=cl, **kwargs)


class Kruskal(SignificanceTest):
    def __init__(self, sig_threshold=0.05, cl=1, **kwargs):
        name = "kruskal"
        super().__init__(
            sfuns.kruskal, False, sig_threshold, name=name,
            cl=cl, **kwargs)


class Friedman(SignificanceTest):
    def __init__(self, sig_threshold=0.05, cl=1, **kwargs):
        name = "friedman"
        super().__init__(
            sfuns.friedman, True, sig_threshold, name=name,
            cl=cl, **kwargs)
