from .ner_span_based_f1 import calculate_span_based_f1_measure
from ..abstract_tests.metric_test import MetricTest
from expanded_checklist.checklist.utils import DataShape, FlattenGroup
from overrides import overrides
import pandas as pd
from munch import Munch
from tabulate import tabulate


class BasicSeqMetrics(MetricTest):
    def __init__(self) -> None:
        """
        The test that calculates a number of different classic metrics for
        classification -- the results from this test can be further used to
        calculate other metrics (e.g. see equality difference).
        """
        super().__init__(
            'seq_metrics', required_ds=DataShape.GROUPED,
            only_accumulate=False, probability_based=False,
            drop_none_labels=True)

    def get_results(self, labels, preds, confs, meta, **kwargs):
        # the get_results function is called on a single group
        # self.only_accumulate is False
        res = calculate_span_based_f1_measure(
            predictions=preds,
            gold_labels=labels,
            # meta allows to handle token mismatches for identity terms
            meta=meta,
            label_encoding="BIOUL")
        return res

    def summary(self, core_record, res_munch, verbose=False, **kwargs):
        results = res_munch.results
        n_examples = res_munch.n_examples

        print(f"Examples used for evaluation: {n_examples}")
        # for group, res in results.items():
        #     res = pd.DataFrame.from_dict(
        #         res, orient='index', columns=res["Tags"])
        #     print(f"\n====== {group}")
        #     print(res)

        for cl in self.classes + ["all"]:
            table = [["Groups:"] + sorted(results.keys())]
            metrics = None
            for gname, stats in results.items():
                # e.g. accumulated might not always appear in the results
                if cl not in stats:
                    continue
                metrics = stats[cl].keys()
                break
            if not metrics:
                continue

            for key in sorted(metrics):
                row = [f"{key}:"]
                for gname in sorted(results.keys()):
                    if cl not in results[gname] or \
                            key not in results[gname][cl]:
                        continue
                    row.append(f"{results[gname][cl][key]:.3f}")
                table.append(row)
            if len(table) > 1:
                print(f"======== CLASS {cl}")
                print(tabulate(table))
