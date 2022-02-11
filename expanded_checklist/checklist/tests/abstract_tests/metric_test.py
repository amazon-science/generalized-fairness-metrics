
from abc import abstractmethod

from .abstract_test import AbstractTest
from munch import Munch
from expanded_checklist.checklist.eval_core import CoreRecord
from expanded_checklist.checklist.utils import DataShape, FlattenGroup

from functools import partial
from typing import Set, Any


class MetricTest(AbstractTest):
    def __init__(
        self,
        name: str,
        required_ds: DataShape,
        only_accumulate: bool = False,
        drop_none_labels: bool = True,
        probability_based: bool = False
    ) -> None:
        """
        Arguments:
            name: name of the test
            only_accumulate: Set to True if the metric doesn't support
                per-group results. In such case the get_results is given data
                for all groups (preds and confs are 2d for classification and
                3d for sequence labeling). If False the get_results function
                gets the data for a single group (preds and confs are 1d for
                classification and 2d for sequence labeling).
        """
        self._name: str = name
        self.only_accumulate: bool = only_accumulate
        self._required_ds: DataShape = required_ds
        self._drop_none_labels = drop_none_labels
        self._probability_based = probability_based

        if self._group_flatten_method is None:
            if required_ds == DataShape.GROUPED:
                self._group_flatten_method = FlattenGroup.FLATTEN
            else:
                self._group_flatten_method = FlattenGroup.RANDOM_MATCH

        self.classes = None
        self.n_classes = None
        self.group_names = None
        self.seq = None

    def compute(
        self,
        core_record: CoreRecord
    ) -> Munch:
        n_examples = core_record.get_n_examples()

        # for seq labelling this gets classes; e.g. PER, ORG, O etc.
        # *not* labels (B-PER, I-PER etc.)
        self.classes = core_record.get_classes()
        self.n_classes: int = len(self.classes)
        self.seq = core_record.sequence
        self.group_names = core_record.group_names

        results = {}

        get_results = partial(
            self.get_results, data_structure=core_record.data_structure)

        # get results for each group independently and then accumulate
        if not self.only_accumulate:
            # to support single-group evaluation the data has to be
            # restructured accordingly
            assert core_record.data_structure == DataShape.GROUPED

            for i, (gname, labels, preds, confs, meta) in enumerate(zip(
                    self.group_names, core_record.labels,
                    core_record.preds, core_record.confs, core_record.meta)):
                results[gname] = get_results(labels, preds, confs, meta)

            # results for *all* groups
            results['all'] = get_results(
                *core_record.get_data_for_many_groups())

            results = self.cross_group_accumulation(results)
        else:
            results = get_results(
                core_record.labels, core_record.preds,
                core_record.confs, core_record.meta)

        return Munch({"results": results, "n_examples": n_examples})

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
        return results

    @abstractmethod
    def get_results(self, labels, preds, confs, meta, **kwargs):
        """
        Get results for one group (if self.only_accumulate is False) or for
        many groups at once (if self.only_accumulate is True).
        """
        raise NotImplementedError

