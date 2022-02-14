from typing import Any
from pandas_ml import ConfusionMatrix
import sklearn.metrics as metrics
from tabulate import tabulate
from collections import defaultdict

from expanded_checklist.checklist.utils import \
    DataShape, ACCUMULATED_STR, FlattenGroup
from ..abstract_tests import ClassificationMetric


class BasicClassificationMetrics(ClassificationMetric):
    def __init__(self) -> None:
        """
        The test that calculates a number of different classic metrics for
        classification -- the results from this test can be further used to
        calculate other metrics (e.g. see equality difference).
        """
        super().__init__('class_metrics', required_ds=DataShape.GROUPED)

    def get_binary_class_results(self, labels, preds, confs):
        # get all metrics from pandas ml confusion matrix
        cm = ConfusionMatrix(labels, preds)
        stats = cm.stats()
        stats["roc_auc"] = metrics.roc_auc_score(labels, preds)
        stats["F1"] = stats["F1_score"]
        return stats

    def cross_class_accumulation(
        self, class_results, labels, preds, confs
    ) -> Any:
        to_accumulate = ["TPR", "PPV", "F1_score"]
        accumulated = defaultdict(list)

        for cl in range(self.n_classes):
            if cl not in class_results:
                continue

            res = class_results[cl]
            for key in to_accumulate:
                if key not in res:
                    continue
                accumulated[key].append(res[key])

        to_ret = {}
        for key, scores in accumulated.items():
            to_ret[f"macro_{key}"] = sum(scores)/len(scores)

        # calculate accuracy on the original labels and preds (all classes)
        to_ret["Accuracy"] = metrics.accuracy_score(labels, preds)
        return to_ret

    def print_for_binary_class(self, results, cl):
        table = [["Groups:"] + sorted(results.keys())]
        metrics = None
        for gname, stats in results.items():
            # e.g. accumulated might not always appear in the results
            if cl not in stats:
                return
            metrics = stats[cl].keys()
            break

        for key in sorted(metrics):
            row = [f"{key}:"]
            for gname in sorted(results.keys()):
                if cl not in results[gname] or key not in results[gname][cl]:
                    continue
                row.append(f"{results[gname][cl][key]:.3f}")
            table.append(row)
        if len(table) > 1:
            print(f"======== CLASS {cl}")
            print(tabulate(table))

    def summary(self, core_record, res_munch, verbose=False, **kwargs):
        results = res_munch.results
        n_examples = res_munch.n_examples

        print(f"Examples used for evaluation: {n_examples}")
        if self.n_classes == 2:
            self.print_for_binary_class(results, 1)
        else:
            print("==== Results accumulates across the classes")
            self.print_for_binary_class(results, ACCUMULATED_STR)

            print("==== Results for one class vs other")
            for i in range(self.n_classes):
                self.print_for_binary_class(results, i)
