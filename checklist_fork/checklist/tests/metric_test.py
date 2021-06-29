
from abc import abstractmethod

from .abstract_test import AbstractTest
from munch import Munch
from checklist_fork.checklist.eval_core import CoreRecord
from checklist_fork.checklist.utils import DataShape, group_data

from typing import List, Union, Dict, Any


class MetricTest(AbstractTest):
    def __init__(
        self,
        name: str,
        required_ds: DataShape,
        only_accumulate: bool = False
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
        self.name: str = name
        self.only_accumulate: bool = only_accumulate
        self.required_ds: DataShape = required_ds

    def get_name(self) -> str:
        return self.name

    def get_group_names(
        self,
        core_record: CoreRecord
    ) -> List[str]:
        """
        Returns the names of all groups, based on either
        (i) the metadata held in the core_record, as ordered in the template
        instantiations. The order is important to determine which scores belong
        to what group.
        or
        (ii) the group_names field in core_record -- this field is set only
        if the data was grouped to start with (had the #ngroups x #nexemples
        structure).

        IMPORTANT: In the (i) case, the implementation assumes the order is the
        same for all templated sentences!
        """
        if core_record.group_names is not None:
            return core_record.group_names

        if core_record.data_structure == DataShape.GROUPED:
            n_groups = len(core_record.preds)
        else:
            n_groups = len(core_record.preds[0])

        if core_record.meta:
            return [core_record.meta[0][x] for x in range(n_groups)]
        else:
            return [f"Group{x+1}" for x in range(n_groups)]

    def compute(
        self,
        core_record: CoreRecord
    ) -> Munch:
        # get names first -- before the restructuring (just in case)
        self.group_names = self.get_group_names(core_record)

        # restructure the data -- this is dependent on the metric/test
        self.process_labels_preds_and_confs(core_record)
        n_examples = self.get_n_examples(core_record)

        results = {}
        # get results for each group independently and then accumulate
        if not self.only_accumulate:
            # to support single-group evaluation the data has to be
            # restructured accordingly
            assert core_record.data_structure == DataShape.GROUPED

            for gname, labels, preds, confs, meta in zip(
                    self.group_names, core_record.labels,
                    core_record.preds, core_record.confs, core_record.meta):

                results[gname] = self.get_results(
                    labels, preds, confs, meta,
                    data_structure=core_record.data_structure)

            results['all'] = self.get_results(
                [x for l in core_record.labels for x in l],
                [x for p in core_record.preds for x in p],
                [x for c in core_record.confs for x in c],
                [x for m in core_record.meta for x in m],
                data_structure=core_record.data_structure)

            results = self.cross_group_accumulation(results)
        else:
            results = self.get_results(
                core_record.labels, core_record.preds, core_record.confs,
                core_record.meta, data_structure=core_record.data_structure)

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

    def process_labels_preds_and_confs(
        self,
        core_record: CoreRecord
    ) -> None:
        """
        (Potentially) restructure the labels, preds and confs stored in the
        core_record. Note: each test gets its own CoreRecord instance, so other
        tests won't be affected by altering fields in core_record.

        Return the number of remaining test examples overall (int) or per
        group (dict).
        """
        if self.required_ds == DataShape.GROUPED:
            if core_record.data_structure != DataShape.GROUPED:
                # #examples x #groups => #groups x #examples
                group_data(core_record)
        elif self.required_ds == DataShape.UNGROUPED and \
                core_record.data_structure == DataShape.GROUPED:
            raise Exception(
                'Cannot run a test for ungrouped/counterfatual data on ' +
                'data split into groups!')
        elif self.required_ds != DataShape.UNGROUPED:
            raise Exception(f'Unsupported data structure: {self.required_ds}')

    def get_n_examples(
        self,
        core_record: CoreRecord
    ) -> Union[int, Dict[str, int]]:
        if self.required_ds == DataShape.GROUPED:
            if all([x == core_record.labels[0] for x in core_record.labels]):
                # all groups have the same number of examples
                return len(core_record.labels[0])
            else:
                # the data must have been originally 'grouped' to allow for a
                # mismatch
                return {k: len(core_record.labels[i])
                        for i, k in enumerate(core_record.group_names)}
        elif self.required_ds == DataShape.UNGROUPED:
            return len(core_record.labels)
        else:
            raise Exception(f'Unsupported data structure: {self.required_ds}')

    @abstractmethod
    def get_results(self, labels, preds, confs, meta, **kwargs):
        """
        Get results for one group (if self.only_accumulate is False) or for
        many groups at once (if self.only_accumulate is True).
        """
        raise NotImplementedError
