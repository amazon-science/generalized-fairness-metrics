from typing import Any, Set, Dict, NewType
from tabulate import tabulate

from expanded_checklist.checklist.utils import DataShape
from expanded_checklist.checklist.eval_core import CoreRecord

from .abstract_test import AbstractTest
from abc import abstractmethod

import expanded_checklist.checklist.tests.helpers as hp
from expanded_checklist.checklist.utils import ACCUMULATED_STR, FlattenGroup

from munch import Munch
from itertools import combinations
from collections import defaultdict

Score = NewType('Score', Any)


class GroupMetric(AbstractTest):
    def __init__(
        self, name: str,
        measure_fun_type: hp.MeasureFunction,
        normalize_fun_type: hp.NormalizeFunction,
        # controls whether examples iwth no labels are filtered out
        group_flatten_method=FlattenGroup.FLATTEN,
        drop_none_labels: bool = True,
        probability_based: bool = False,
        vector_valued: bool = False
    ):
        self._name: str = name
        self._required_ds = DataShape.GROUPED
        self._drop_none_labels = drop_none_labels
        self._probability_based = probability_based
        self._group_flatten_method = group_flatten_method

        self.classes = None
        self.n_classes = None
        self.group_names = None

        self.vector_valued = vector_valued
        self.measure_gap = hp.get_measure_fun(measure_fun_type)
        self.normalize = hp.get_normalize_fun(normalize_fun_type)

    def compute(
        self,
        core_record: CoreRecord
    ) -> Munch:
        n_examples = core_record.get_n_examples()

        self.group_names = core_record.group_names
        self.classes = core_record.get_classes()
        self.n_classes: int = len(self.classes)
        self.seq = core_record.sequence

        assert core_record.data_structure == self.required_ds

        self.group_results =\
            self.get_scores_for_relevant_example_sets(core_record)
        results = self.cross_group_accumulation(self.group_results)
        return Munch({"results": results, "n_examples": n_examples})

    def summary(self, core_record, res_munch, verbose=False, **kwargs):
        results = res_munch.results
        n_examples = res_munch.n_examples

        print(f"Examples used for evaluation: {n_examples}")
        print(f"Groups: {' '. join(self.group_names)}")

        if not self.vector_valued:
            table = [["CLASS", self.get_name()]]
            for cl, name2res in results.items():
                res = name2res[ACCUMULATED_STR]
                row = [cl, f"{res:.3f}"]
                table.append(row)
            print(tabulate(table))
        else:
            header = ["CLASS"]
            table = []
            for i, (cl, name2res) in enumerate(results.items()):
                row = [cl]
                for gname, res in name2res.items():
                    row.append(f"{res:.3f}")
                    if i == 0:
                        header.append(gname)
                table.append(row)
            print(tabulate([header] + table))

    @abstractmethod
    def get_group_score(
            self, labels, preds, confs, meta, **kwargs) -> Dict[int, Score]:
        """
        Returns a class2score map. If there is a single score per all classes,
        it's returned under key 'all'. This is left as an abstract class
        (handled differently from background, measure gap and normalize funs)
        to allow the scores to be more elaborate and metric specific.
        """
        raise NotImplementedError

    @abstractmethod
    def get_scores_for_relevant_example_sets(self, core_record):
        raise NotImplementedError

    @abstractmethod
    def cross_group_accumulation(self, results) -> Any:
        """
        Accumulate the results obtained independently for each group.
        This is only called if self.only_accumulate is False.

        Return the *full dictionary of results* (e.g. the one that was passed
        as a parameted with some additional keys):
        The function supports complete replacement of results -- e.g. if the
        results for individual groups are calculated only in order to be
        accumulated into the final result)
        """
        raise NotImplementedError


class CounterfactualMetric(GroupMetric):
    def __init__(
            self, *args,
            group_flatten_method=FlattenGroup.RANDOM_MATCH, **kwargs):
        super().__init__(*args, **kwargs)
        self._required_ds = DataShape.UNGROUPED
        self._group_flatten_method = group_flatten_method

    def compute(
        self,
        core_record: CoreRecord
    ) -> Munch:
        n_examples = core_record.get_n_examples()

        self.group_names = core_record.group_names
        self.classes = core_record.get_classes()
        self.n_classes: int = len(self.classes)
        self.seq = core_record.sequence

        assert core_record.data_structure == self.required_ds

        results = {}

        assert len(core_record.labels) == n_examples

        # iterate over counterfactual examples
        for labels, preds, confs, meta in zip(
                core_record.labels, core_record.preds,
                core_record.confs, core_record.meta):

            group_results =\
                self.get_scores_for_relevant_example_sets(
                    labels, preds, confs, meta)

            ex_results = self.cross_group_accumulation(group_results)

            for cl, name2res in ex_results.items():
                for name, res in name2res.items():
                    # accumulate towards average over all examples
                    if cl not in results:
                        results[cl] = defaultdict(int)
                    results[cl][name] += res/n_examples

        return Munch({"results": results, "n_examples": n_examples})

    @abstractmethod
    def get_scores_for_relevant_example_sets(
            self, ex_labels, ex_preds, ex_confs, ex_meta):
        raise NotImplementedError


class BCM(GroupMetric):
    """
    Class defined for convenience, to avoid code repetition in
    counterfactual and group versions for the metrics.
    """
    def __init__(
        self, name: str,
        measure_fun_type: hp.MeasureFunction,
        normalize_fun_type: hp.NormalizeFunction,
        background_type: hp.Background,
        **kwargs
    ):
        super().__init__(
            name, measure_fun_type, normalize_fun_type, **kwargs)
        self.get_background = hp.get_background_fun(background_type)

    def get_background_score(
            self, labels, preds, confs, meta, **kwargs) -> Dict[int, Score]:
        # this can be overriden if the background score should be computed
        # differently to the group score
        return self.get_group_score(labels, preds, confs, meta, **kwargs)

    def cross_group_accumulation(self, results) -> Any:
        classes = self.classes

        new_results = {}
        #print(self.name)
        for cl in classes + ["all"]:
            final_gap = 0

            # missing results for one (or more) of the groups for that class
            # - skip the class
            if any([cl not in score for _, score in results.items()]):
                continue

            new_results[cl] = {}
            for gname, group_score in results.items():
                if gname not in self.group_names:
                    continue

                background_score = results[f'{gname}_background'][cl]
                group_score = group_score[cl]

                gap = self.measure_gap(background_score, group_score)

                #print(gname)
                #print("score:", group_score, "bg:", background_score, "gap:", gap)
                new_results[cl][gname] = gap
                final_gap += gap

            final_gap = self.normalize(final_gap, len(self.group_names))
            new_results[cl][ACCUMULATED_STR] = final_gap
        return new_results


class PCM(GroupMetric):
    """
    Class defined for convenience, to avoid code repetition in
    counterfactual and group versions for the metrics.
    """
    def cross_group_accumulation(self, results) -> Any:
        classes = self.classes

        new_results = {}
        for cl in classes + ["all"]:
            final_gap = 0

            # missing results for one (or more) of the groups for that class
            # - skip the class
            if any([cl not in score for _, score in results.items()]):
                continue

            new_results[cl] = {}
            for gname1, gname2 in combinations(self.group_names, 2):
                g1_score = results[gname1][cl]
                g2_score = results[gname2][cl]
                gap = self.measure_gap(g1_score, g2_score)

                new_results[cl][(gname1, gname2)] = gap
                final_gap += gap

            for gname in self.group_names:
                all_scores =\
                    [score for k, score in new_results[cl].items()
                     if gname in set(k)]
                new_results[cl][gname] = sum(all_scores)/len(all_scores)

            final_gap = self.normalize(final_gap, len(self.group_names))
            new_results[cl][ACCUMULATED_STR] = final_gap
        return new_results


class MCM(GroupMetric):
    """
    Class defined for convenience, to avoid code repetition in
    counterfactual and group versions for the metrics.
    """
    def __init__(
        self, name: str,
        measure_fun_type: hp.MeasureFunction, **kwargs
    ):
        super().__init__(
            name, measure_fun_type,
            normalize_fun_type=hp.NormalizeFunction.NONE, **kwargs)

    def cross_group_accumulation(self, results) -> Any:
        classes = self.classes

        new_results = {}
        for cl in classes + ["all"]:
            final_gap = 0

            if any([cl not in score for _, score in results.items()]):
                continue

            new_results[cl] = {}
            group_scores = [results[gname][cl] for gname in self.group_names]
            gap = self.measure_gap(group_scores)

            final_gap = self.normalize(gap, len(self.group_names))
            new_results[cl][ACCUMULATED_STR] = final_gap
        return new_results


################

class GroupBCM(BCM):
    def get_scores_for_relevant_example_sets(self, core_record):
        results = {}

        for i, (gname, labels, preds, confs, meta) in enumerate(zip(
                self.group_names, core_record.labels,
                core_record.preds, core_record.confs, core_record.meta)):

            results[gname] = self.get_group_score(labels, preds, confs, meta)

            bg = self.get_background(
                core_record.labels, core_record.preds,
                core_record.confs, core_record.meta, gindex=i, gname=gname)
            results[f'{gname}_background'] = self.get_background_score(*bg)
        return results


class GroupPCM(PCM):
    def get_scores_for_relevant_example_sets(self, core_record):
        results = {}
        for i, (gname, labels, preds, confs, meta) in enumerate(zip(
                self.group_names, core_record.labels,
                core_record.preds, core_record.confs, core_record.meta)):
            results[gname] = self.get_group_score(labels, preds, confs, meta)
        return results


class GroupMCM(MCM):
    def get_scores_for_relevant_example_sets(self, core_record):
        results = {}
        for i, (gname, labels, preds, confs, meta) in enumerate(zip(
                self.group_names, core_record.labels,
                core_record.preds, core_record.confs, core_record.meta)):
            results[gname] = self.get_group_score(labels, preds, confs, meta)
        return results


class CounterfactualBCM(BCM, CounterfactualMetric):
    def get_scores_for_relevant_example_sets(
            self, ex_labels, ex_preds, ex_confs, ex_meta):
        results = {}

        for i, (gname, labels, preds, confs, meta) in enumerate(zip(
                self.group_names, ex_labels, ex_preds, ex_confs, ex_meta)):

            # in many scenarios labels, preds and confs can be scalars
            # instead of lists (if there is only one cf example per group)
            results[gname] = self.get_group_score(labels, preds, confs, meta)

            bg = self.get_background(
                ex_labels, ex_preds, ex_confs, ex_meta,
                gindex=i, gname=gname)
            results[f'{gname}_background'] = self.get_background_score(*bg)
        return results


class CounterfactualPCM(PCM, CounterfactualMetric):
    def get_scores_for_relevant_example_sets(
            self, ex_labels, ex_preds, ex_confs, ex_meta):
        results = {}
        labels = ex_labels
        meta = ex_meta
        for i, (gname, preds, confs) in enumerate(zip(
                self.group_names, ex_preds, ex_confs)):
            # in many scenarios labels, preds and confs can be scalars
            # instead of lists (if there is only one cf example per group)
            results[gname] = self.get_group_score(labels, preds, confs, meta)
        return results


class CounterfactualMCM(MCM, CounterfactualMetric):
    def get_scores_for_relevant_example_sets(
            self, ex_labels, ex_preds, ex_confs, ex_meta):
        results = {}
        labels = ex_labels
        meta = ex_meta
        for i, (gname, preds, confs) in enumerate(zip(
                self.group_names, ex_preds, ex_confs)):
            # in many scenarios labels, preds and confs can be scalars
            # instead of lists (if there is only one cf example per group)
            results[gname] = self.get_group_score(labels, preds, confs, meta)
        return results
