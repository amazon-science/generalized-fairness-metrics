from .abstract_tests import AbstractTest

from typing import List
from ..eval_core import CoreRecord
from munch import Munch
from tabulate import tabulate

from expanded_checklist.checklist.utils import ACCUMULATED_STR
from copy import deepcopy


class MetricBundle(AbstractTest):
    def __init__(self, name, metrics: List):
        """
        This class can be used to bundle a number of metrics which return
        a single value per class; the results field in the munch they return
        for the compute function is a class2scalar map.
        It is not suitable to bundle metrics which return  multiple values.
        """
        self._name = name
        self.mname2metric = {m.get_name(): m for m in metrics}

    def compute(self, core_record: CoreRecord):
        mname2res = {}
        for mname, m in self.mname2metric.items():
            mname2res[mname] = m.compute(deepcopy(core_record))
        return Munch({'results': mname2res})

    def summary(self, core_record, res_munch, verbose=False, **kwargs):
        mname2res = res_munch.results

        table = []
        header = ["Metric", "#examples"]
        group_names = None
        classes = None

        for mname in sorted(mname2res.keys()):
            m = self.mname2metric[mname]
            res = mname2res[mname]
            results = res.results
            n_examples = res.n_examples

            mrow = [mname, n_examples]
            if not group_names:
                group_names = m.group_names
                classes = sorted(results.keys())
            else:
                assert group_names == m.group_names
                assert sorted(results.keys()) == classes

            for cl in classes:
                res = results[cl][ACCUMULATED_STR]
                mrow.append(f"{res:.3f}")
            table.append(mrow)

        print(f"Groups: {' '. join(group_names)}")
        header += classes
        print(tabulate([header] + table))
