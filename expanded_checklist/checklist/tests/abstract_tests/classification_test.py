from abc import abstractmethod
from typing import Any, Dict
import pandas as pd
from .metric_test import MetricTest
from overrides import overrides
from munch import Munch
from expanded_checklist.checklist.utils import \
    DataShape, is_2d_list, ACCUMULATED_STR
pd.options.display.float_format = "{:,.2f}".format


class ClassificationMetric(MetricTest):
    def __init__(
        self,
        name: str,
        required_ds: DataShape,
        only_accumulate: bool = False
    ) -> None:
        """
        Arguments:
            name: name of the test

            For the remaining parameters see MetricTest
        """
        super().__init__(
            name, required_ds, only_accumulate, probability_based=False,
            drop_none_labels=True)

    @abstractmethod
    def get_binary_class_results(self, labels, preds, confs) -> Any:
        """
        Get results for the binary classification.
        """
        raise NotImplementedError

    def get_results(self, labels, preds, confs, meta, **kwargs):
        """
        Get one vs other results for each class.
        If a metric/test supports direct calculation of multi-class results
        then it should override this function or (even better) the get_results
        function (which calls this function).
        """
        # classes are ints, starting from 0
        class_results = {}

        if 'n_classes' in kwargs:
            self.n_classes = kwargs['n_classes']

        for cl in range(self.n_classes):
            # turn the data into one vs other
            if is_2d_list(labels):
                labels_tmp = []
                for x in labels:
                    labels_tmp.append([True if i == cl else False for i in x])
                    if not any(labels_tmp[-1]):
                        continue
            else:
                labels_tmp = [True if x == cl else False for x in labels]
                if not any(labels_tmp):
                    continue

            if not is_2d_list(preds):
                pred_tmp = [True if x == cl else False for x in preds]
                # get the conf score for a particular class
                conf_tmp = [x[cl] for x in confs]
            else:
                pred_tmp = []
                conf_tmp = []
                for x in preds:
                    pred_tmp.append([True if i == cl else False for i in x])
                # get the conf score for a particular class
                for x in confs:
                    conf_tmp.append([i[cl] for i in x])

            res = self.get_binary_class_results(labels_tmp, pred_tmp, conf_tmp)
            class_results[cl] = res

        accumulated = self.cross_class_accumulation(
            class_results, labels, preds, confs)
        if accumulated:
            class_results[ACCUMULATED_STR] = accumulated
        return class_results

    def cross_class_accumulation(
        self,
        class_results: Dict,  # class to results mapping
        labels, preds, confs
    ) -> Any:
        """
        Accumulate the results obtained for each class independently.

        Return the result for all classes, to be assigned as a value for
        ACCUMULATED_STR key.
        """
        return {}
