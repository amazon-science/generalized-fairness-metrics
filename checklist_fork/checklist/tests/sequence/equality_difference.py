from munch import Munch
from typing import Any
from .basic_seq_metrics import BasicSeqMetrics
from tabulate import tabulate


class SeqEqualityDifference(BasicSeqMetrics):
    def __init__(self):
        super().__init__()
        # override the name
        self._name = 'equality_difference'

    def cross_group_accumulation(self, results) -> Any:
        fnr_str = "FNR"
        full_results = results['all']
        classes = sorted(full_results.keys())

        new_results = {}
        for cl in classes:
            FNED = 0
            # missing results for one (or more) of the groups for that class
            # - skip the class
            if any([cl not in stats for _, stats in results.items()]):
                continue

            all_res = full_results[cl]
            all_fnr = all_res[fnr_str]

            for gname, stats in results.items():
                if gname == "all":
                    continue

                fnr = stats[cl][fnr_str]
                FNED += abs(all_fnr - fnr)
            new_results[cl] = Munch({"fned": FNED})
        return new_results

    def summary(self, core_record, res_munch, verbose=False, **kwargs):
        results = res_munch.results
        n_examples = res_munch.n_examples

        print(f"Examples used for evaluation: {n_examples}")
        print(f"Groups: {' '. join(self.group_names)}")
        table = [["CLASS", "FNED"]]
        for cl, res_munch in results.items():
            row = [cl, f"{res_munch.fned:.3f}"]
            table.append(row)
        print(tabulate(table))
