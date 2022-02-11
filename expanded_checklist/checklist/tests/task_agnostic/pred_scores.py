from ..abstract_tests import GroupBCM, GroupMetric, GroupPCM, GroupMCM
import expanded_checklist.checklist.tests.helpers as hp
from expanded_checklist.checklist.tests import BasicClassificationMetrics
from expanded_checklist.checklist.tests import BasicSeqMetrics

from expanded_checklist.checklist.utils import is_1d_list, is_2d_list, FlattenGroup

import itertools
from ..metric_bundle import MetricBundle


###################################
#  SCORE CALCULATORS (helpers)    #
###################################

class BasicMetricScore(GroupMetric):
    def __init__(
        self,
        score_str
    ):
        """
        Arguments:
            name - name of the metrics
            score_str - name of the score to use in metric calculations (e.g.
            'FNR' or 'F1'. This string has to be present in the results dict
            for obtained for each group (check get_results for
            SeqBasicMetrics or BasicClassificationMetrics).
        """
        self.basic_seq_metrics = BasicSeqMetrics()
        self.basic_class_metrics = BasicClassificationMetrics()
        self.score_str = score_str

    def get_group_score(self, labels, preds, confs, meta, **kwargs):
        if not self.seq:
            metric_results = self.basic_class_metrics.get_results(
                labels, preds, confs, meta, n_classes=self.n_classes)
        else:
            metric_results = self.basic_seq_metrics.get_results(
                labels, preds, confs, meta, n_classes=self.n_classes)

        to_ret = {}
        for cl, res in metric_results.items():
            if cl not in self.classes:
                continue

            if self.score_str not in res:
                #print(res.keys())
                raise Exception(f'{self.score_str} not available for the ' +
                                f'task, seq:{self.seq}')
            to_ret[cl] = res[self.score_str]
        return to_ret


class BasicMetricBCM(BasicMetricScore, GroupBCM):
    def __init__(self, name: str, score_str: str, *args, **kwargs):
        BasicMetricScore.__init__(self, score_str)
        GroupBCM.__init__(self, name, *args, **kwargs)


class BasicMetricPCM(BasicMetricScore, GroupPCM):
    def __init__(self, name: str, score_str: str, *args, **kwargs):
        BasicMetricScore.__init__(self, score_str)
        GroupPCM.__init__(self, name, *args, **kwargs)


class BasicMetricMCM(BasicMetricScore, GroupMCM):
    def __init__(self, name: str, score_str: str, *args, **kwargs):
        BasicMetricScore.__init__(self, score_str)
        GroupMCM.__init__(self, name, *args, **kwargs)

########################
#  EXISTING METRICS    #
########################


#   FNED & FNED (Dixon et al. 2018)

class FNED(BasicMetricBCM):
    def __init__(self):
        super().__init__(
            'fned', 'FNR',
            hp.MeasureFunction.ABS_DIFF, hp.NormalizeFunction.NONE,
            hp.Background.ALL_GROUPS, vector_valued=True)


class FPED(BasicMetricBCM):
    def __init__(self):
        super().__init__(
            'fped', 'FPR',
            hp.MeasureFunction.ABS_DIFF, hp.NormalizeFunction.NONE,
            hp.Background.ALL_GROUPS, vector_valued=True)


# normalized correctly

class FixedFNED(BasicMetricBCM):
    def __init__(self):
        super().__init__(
            'fixed_fned', 'FNR',
            hp.MeasureFunction.ABS_DIFF, hp.NormalizeFunction.NGROUPS,
            hp.Background.ALL_GROUPS, vector_valued=True)


class FixedFPED(BasicMetricBCM):
    def __init__(self):
        super().__init__(
            'fixed_fped', 'FPR',
            hp.MeasureFunction.ABS_DIFF, hp.NormalizeFunction.NGROUPS,
            hp.Background.ALL_GROUPS, vector_valued=True)


#   FPR Gap (Beutel et al. 2019) (and FNR equivalent)

class FPRRatioBCM(BasicMetricBCM):
    def __init__(self):
        super().__init__(
            'fpr_ratio_bcm', 'FPR',
            # right -- group/background
            hp.MeasureFunction.RATIO_RIGHT, hp.NormalizeFunction.NGROUPS,
            hp.Background.EXCLUDE_GROUP, vector_valued=True)


class FNRRatioBCM(BasicMetricBCM):
    def __init__(self):
        super().__init__(
            'fnr_ratio_bcm', 'FNR',
            # right -- group/background
            hp.MeasureFunction.RATIO_RIGHT, hp.NormalizeFunction.NGROUPS,
            hp.Background.EXCLUDE_GROUP, vector_valued=True)


#   True Positive/Negative Rate Gap (Prost et al. 2019)

class TPRGapPCM(BasicMetricPCM):
    def __init__(self):
        super().__init__(
            'tpr_gap_pcm', 'TPR',
            hp.MeasureFunction.ABS_DIFF, hp.NormalizeFunction.NPAIRS
        )


class TNRGapPCM(BasicMetricPCM):
    def __init__(self):
        super().__init__(
            'tnr_gap_pcm', 'TNR',
            hp.MeasureFunction.ABS_DIFF, hp.NormalizeFunction.NPAIRS
        )


#   Disparsity Score (Gaut et al. 2020)

class DisparsityScore(BasicMetricPCM):
    def __init__(self):
        super().__init__(
            'disparsity_score', 'F1',
            hp.MeasureFunction.ABS_DIFF, hp.NormalizeFunction.NGROUPS)


class FixedDisparsityScore(BasicMetricPCM):
    def __init__(self):
        super().__init__(
            'fixed_disparsity_score', 'F1',
            hp.MeasureFunction.ABS_DIFF, hp.NormalizeFunction.NPAIRS)


#   Parity Gap
class ParityGap(GroupPCM):
    def __init__(self):
        self.basic_seq_metrics = BasicSeqMetrics()
        self.basic_class_metrics = BasicClassificationMetrics()

        super().__init__(
            'parity_gap', hp.MeasureFunction.ABS_DIFF,
            hp.NormalizeFunction.NPAIRS)

    def get_group_score(self, labels, preds, confs, meta, **kwargs):
        if not self.seq:
            metric_results = self.basic_class_metrics.get_results(
                labels, preds, confs, meta, n_classes=self.n_classes)
        else:
            metric_results = self.basic_seq_metrics.get_results(
                labels, preds, confs, meta, n_classes=self.n_classes)

        to_ret = {}
        for cl, res in metric_results.items():
            if cl not in self.classes:
                continue

            if "TP" not in res or "FP" not in res:
                #print(res.keys())
                raise Exception(f'TP or FP not available for the ' +
                                f'task, seq:{self.seq}')

            tp = res["TP"]
            fp = res["FP"]

            # e.g. the probability of predicting positive
            to_ret[cl] = (tp + fp)/len(labels)
        return to_ret


# # adapted as PCM
# class PerturbationLabelDistance(GroupPCM):
#     def __init__(self):
#         super().__init__(
#             'perturbation_label_dist', hp.MeasureFunction.JACCARD_DIST,
#             hp.NormalizeFunction.NPAIRS,
#             group_flatten_method=FlattenGroup.AVERAGE
#         )

#     def get_group_score(self, labels, preds, confs, meta, **kwargs):
#         to_ret = {}
#         for cl in range(self.n_classes):
#             # turn preds into one vs other
#             if not is_2d_list(preds):
#                 pred_tmp = [True if x == cl else False for x in preds]
#             else:
#                 pred_tmp = []
#                 for x in preds:
#                     pred_tmp.append([True if i == cl else False for i in x])

#             to_ret[cl] = pred_tmp
#         return to_ret

#########################################
#   CREATE ALL OPTIONS FOR GIVEN SCORE  #
#########################################

def get_all_versions_of_the_group_metric(name, score_str):
    metrics = []

    # GET BCM
    for bg, mf, nf in itertools.product(
            hp.Background,
            hp.get_scalar_measure_funs(),
            [hp.NormalizeFunction.NONE, hp.NormalizeFunction.NGROUPS]):
        tmp_name =\
            f"BCM_{name}_{bg.name.lower()}-{mf.name.lower()}-{nf.name.lower()}"
        tmp_metric = BasicMetricBCM(tmp_name, score_str, mf, nf, bg)
        metrics.append(tmp_metric)

    # GET PCM
    for mf, nf in itertools.product(
            hp.get_scalar_measure_funs(),
            [hp.NormalizeFunction.NONE, hp.NormalizeFunction.NPAIRS]):
        tmp_name =\
            f"PCM_{name}_{mf.name.lower()}-{nf.name.lower()}"
        tmp_metric = BasicMetricPCM(
            tmp_name, score_str, mf, nf, vector_valued=False)
        metrics.append(tmp_metric)

    return MetricBundle(f'{name}_bundle', metrics)


####################
#   NEW METRICS    #
####################

class FNRRange(BasicMetricMCM):
    def __init__(self):
        super().__init__(
            'fnr_range', 'FNR', hp.MeasureFunction.RANGE,
            drop_none_labels=True,
            probability_based=False
        )


class FPRRange(BasicMetricMCM):
    def __init__(self):
        super().__init__(
            'fpr_range', 'FPR', hp.MeasureFunction.RANGE,
            drop_none_labels=True,
            probability_based=False
        )
