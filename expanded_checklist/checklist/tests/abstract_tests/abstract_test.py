from abc import ABC, abstractmethod
from expanded_checklist.checklist.core_record import CoreRecord
from expanded_checklist.checklist.utils import FlattenGroup, DataShape
from munch import Munch
import pandas as pd

pd.options.display.float_format = "{:,.2f}".format


class AbstractTest(ABC):
    # the inheriting classes should define all of the following variables
    _name, _drop_none_labels, \
        _required_ds, _probability_based, _group_flatten_method = [None] * 5

    @property
    def name(self) -> str:
        return self._name

    @property
    def drop_none_labels(self) -> bool:
        return self._drop_none_labels

    @property
    def required_ds(self) -> DataShape:
        return self._required_ds

    @property
    def probability_based(self) -> bool:
        return self._probability_based

    @property
    def group_flatten_method(self) -> FlattenGroup:
        return self._group_flatten_method

    @abstractmethod
    def compute(
        self,
        core_record: CoreRecord
    ) -> Munch:
        """
        Get the results for that test on the data and predictions stored in
        the given CoreRecord instance. The results should be returned as a
        munch (typicall holding 'results' field and some extra fields.)

        This function is called by the EvaluationCore instance to get eval
        results for that test/metric. Each test gets its own instance of
        CoreRecord so it can freely alter all it's fields as it computes the
        metrics.
        """
        raise NotImplementedError

    @abstractmethod
    def summary(
        self,
        core_record: CoreRecord,
        res_munch: Munch,
        verbose=False,
        **kwargs
    ) -> None:
        """
        Print out the results from the test run (results Munc) which were
        computed for the the data and predictions stored in the given
        CoreRecord instance.
        """
        raise NotImplementedError

    def get_name(self) -> str:
        return self.name

    def visualize(
        self,
        core_record: CoreRecord,
        results: Munch,
        **kwargs
    ) -> None:
        raise NotImplementedError
