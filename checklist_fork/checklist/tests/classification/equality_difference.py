from munch import Munch
from typing import Any
from tabulate import tabulate
from .basic_classification_metrics import BasicClassificationMetrics


class EqualityDifference(BasicClassificationMetrics):
    def __init__(self):
        # this class relies on BasicMetric functions to get all metric scores
        # for different classes
        super().__init__()
        # override the name
        self._name = 'equality_difference'

    def cross_group_accumulation(self, results) -> Any:
        fnr_str, fpr_str = "FNR", "FPR"

        classes = [i for i in range(self.n_classes)]
        if len(classes) == 2:
            classes = [1]

        new_results = {}
        for cl in classes:
            FPED, FNED = 0, 0
            # missing results for one (or more) of the groups for that class
            # - skip the class
            if any([cl not in stats for _, stats in results.items()]):
                continue

            all_res = results['all'][cl]
            all_fnr, all_fpr = all_res[fnr_str], all_res[fpr_str]

            for gname, stats in results.items():
                if gname == "all":
                    continue
                fpr, fnr = stats[cl][fpr_str], stats[cl][fnr_str]
                FPED += abs(all_fpr - fpr)
                FNED += abs(all_fnr - fnr)
            new_results[cl] = Munch({"fped": FPED, "fned": FNED})
        return new_results

    def summary(self, core_record, res_munch, verbose=False, **kwargs):
        results = res_munch.results
        n_examples = res_munch.n_examples

        print(f"Examples used for evaluation: {n_examples}")
        print(f"Groups: {' '. join(self.group_names)}")
        table = [["CLASS", "FPED", "FNED"]]
        for cl, res_munch in results.items():
            row = [cl, f"{res_munch.fped:.3f}", f"{res_munch.fned:.3f}"]
            table.append(row)
        print(tabulate(table))
