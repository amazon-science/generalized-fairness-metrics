from abc import ABC, abstractmethod
from ..eval_core import CoreRecord
from munch import Munch
import pandas as pd

pd.options.display.float_format = "{:,.2f}".format


class AbstractTest(ABC):
    # each test needs to have a self.name
    name = "abstract_test"

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

    # @abstractmethod
    # def save_summary(
    #     self,
    #     core_record: CoreRecord,
    #     results: Munch,
    #     **kwargs
    # ) -> None:
    #     """
    #     Saves the summary from the test run on the data and predictions
    #     stored in the given CoreRecord instance.

    #     TODO: This function hasn't been implemented just yet.
    #     """
    #     raise NotImplementedError
