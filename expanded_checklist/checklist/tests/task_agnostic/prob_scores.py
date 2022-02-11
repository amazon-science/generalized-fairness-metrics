from ..abstract_tests import GroupBCM, GroupMetric, GroupPCM, \
    CounterfactualPCM, CounterfactualMCM
import expanded_checklist.checklist.tests.helpers as hp
from expanded_checklist.checklist.utils import is_1d_list, FlattenGroup


###################################
#  SCORE CALCULATORS (helpers)    #
###################################


def check_probs(confs, n_classes, seq):
    if not seq:
        if is_1d_list(confs):
            assert len(confs) == n_classes
        else:
            assert all(len(x) == n_classes for x in confs)
    else:
        for example_confs in confs:
            if is_1d_list(example_confs):
                assert len(example_confs) == n_classes
            else:
                assert all(len(x) == n_classes for x in example_confs)


class ProbabilityScore(GroupMetric):
    """
    For each class return a vector of confidence scores for that class *across
    all examples* (labels are not required).
    """
    def __init__(self, use_pos_score_in_binary_sent: bool = False):
        # some metrics use the prob for the positive class, also
        # for negative examples (e.g. in binary sent classification)
        self.use_pos_score_in_binary_sent = use_pos_score_in_binary_sent

    def get_group_score(self, labels, preds, confs, meta, **kwargs):
        to_ret = {}

        if len(self.classes) != 2 and self.use_pos_score_in_binary_sent:
            raise Exception("use_pos_score_in_binary_sent is True but " +
                            "classification is not binary.")

        check_probs(confs, len(self.classes), self.seq)

        for cl_idx, cl in enumerate(self.classes):
            if self.use_pos_score_in_binary_sent:
                cl_idx = 1

            if not self.seq:
                if is_1d_list(confs):
                    cl_confs = confs[cl_idx]
                else:
                    cl_confs = [ex_confs[cl_idx] for ex_confs in confs]
            else:
                cl_confs = []
                for example_confs in confs:
                    if is_1d_list(example_confs):
                        cl_confs.append(example_confs[cl_idx])
                    else:
                        cl_confs += [
                            tok_confs[cl_idx] for tok_confs in example_confs]
            to_ret[cl] = cl_confs
        return to_ret


class ProbabilityScoreKeepOnlyClass(GroupMetric):
    """
    For each class return a vector of confidence scores for that class,
    *only* for examples that belong to that class (i.e. their gold label is
    that class). Labels are required.
    """
    def __init__(self, use_pos_score_in_binary_sent: bool = False):
        # some metrics use the prob for the positive class, also
        # for negative examples (e.g. in binary sent classification)
        self.use_pos_score_in_binary_sent = use_pos_score_in_binary_sent

    def get_group_score(self, labels, preds, confs, meta, **kwargs):
        to_ret = {}

        if len(self.classes) != 2 and self.use_pos_score_in_binary_sent:
            raise Exception("use_pos_score_in_binary_sent is True but " +
                            "classification is not binary.")

        check_probs(confs, len(self.classes), self.seq)

        if self.group_flatten_method == FlattenGroup.NONE:
            labels = [labels] * len(confs)

        for cl_idx, cl in enumerate(self.classes):
            if self.use_pos_score_in_binary_sent:
                cl_idx = 1

            if not self.seq:
                if is_1d_list(confs):
                    if labels == cl:
                        cl_confs = confs[cl_idx]
                    else:
                        cl_confs = None
                else:
                    if not is_1d_list(labels):
                        labels = [labels] * len(confs)

                    cl_confs = [
                        ex_confs[cl_idx] for i, (ex_confs)
                        in enumerate(confs) if labels[i] == cl]
            else:
                cl_confs = []

                for example_confs, example_labels in zip(confs, labels):
                    if is_1d_list(example_confs):
                        if example_labels == cl:
                            cl_confs.append(example_confs[cl_idx])
                    else:
                        if not is_1d_list(example_labels):
                            example_labels =\
                                [example_labels] * len(example_confs)
                        cl_confs += [
                            tok_confs[cl_idx] for i, tok_confs
                            in enumerate(example_confs)
                            if example_labels[i] == cl]
            if cl_confs:
                to_ret[cl] = cl_confs
        return to_ret


class ProbabilityScoreTarget(GroupMetric):
    """
    For each example return a confidence score associated with its gold label.
    Do not return class-specific scores.
    """
    def get_group_score(self, labels, preds, confs, meta, **kwargs):
        to_ret = {}

        check_probs(confs, len(self.classes), self.seq)

        if not self.seq:
            # just one example
            if is_1d_list(confs):
                target_idx = self.classes.index(labels)
                cl_confs = confs[target_idx]
            else:
                cl_confs = [ex_confs[self.classes.index(ex_label)]
                            for ex_confs, ex_label
                            in zip(confs, labels)]
        else:
            cl_confs = []
            for example_confs, example_labels in zip(confs, labels):
                if is_1d_list(example_confs):
                    cl_confs.append(
                        example_confs[self.classes.index(example_labels)])
                else:
                    cl_confs.append(
                        [tok_confs[self.classes.index(tok_label)] for
                         tok_confs, tok_label
                         in zip(example_confs, example_labels)])
        to_ret["all"] = cl_confs
        return to_ret


########################
#  EXISTING METRICS    #
########################


#   Garg et al. 2019 (adapted as PCM to fit templated scenario)

class CounterfactualFairnessGap(ProbabilityScore, CounterfactualPCM):
    def __init__(self):
        # according to the paper use_pos_score_in_binary_sent=True,
        # but in binary case it's equivalent to False
        ProbabilityScore.__init__(self, use_pos_score_in_binary_sent=False)
        CounterfactualPCM.__init__(
            self,
            "cfgap", hp.MeasureFunction.ABS_DIFF,
            hp.NormalizeFunction.NPAIRS,
            drop_none_labels=False,
            probability_based=True
        )


class CounterfactualFairnessGapOnlyClass(ProbabilityScoreKeepOnlyClass, CounterfactualPCM):
    def __init__(self):
        ProbabilityScoreKeepOnlyClass.__init__(self, False)
        CounterfactualPCM.__init__(
            self,
            "cfgap_only_target_class", hp.MeasureFunction.ABS_DIFF,
            hp.NormalizeFunction.NPAIRS,
            drop_none_labels=True,
            probability_based=True
        )


# Huang et al. (2019)

class AverageGroupFairness(ProbabilityScore, GroupBCM):
    def __init__(self):
        # technically Huang et al. have use_pos_score_in_binary_sent, but
        # they just compare distributions, so the scores are equivalent
        # if one uses p or 1 - p
        ProbabilityScore.__init__(self, use_pos_score_in_binary_sent=False)
        GroupBCM.__init__(
            self,
            "average_group_fairness", hp.MeasureFunction.W1,
            hp.NormalizeFunction.NGROUPS,
            hp.Background.ALL_GROUPS,
            drop_none_labels=False,
            probability_based=True,
            vector_valued=True
        )


class AverageGroupFairnessOnlyClass(ProbabilityScoreKeepOnlyClass, GroupBCM):
    def __init__(self):
        # technically Huang et al. have use_pos_score_in_binary_sent, but
        # they just compare distributions, so the scores are equivalent
        # if one uses p or 1 - p
        ProbabilityScoreKeepOnlyClass.__init__(
            self, use_pos_score_in_binary_sent=False)
        GroupBCM.__init__(
            self,
            "average_group_fairness_only_target_class", hp.MeasureFunction.W1,
            hp.NormalizeFunction.NGROUPS,
            hp.Background.EXCLUDE_GROUP,
            drop_none_labels=True,
            probability_based=True,
            vector_valued=True
        )


class AverageIndividualFairness(ProbabilityScore, CounterfactualPCM):
    def __init__(self):
        ProbabilityScore.__init__(self, use_pos_score_in_binary_sent=False)
        CounterfactualPCM.__init__(
            self,
            "average_individual_fairness", hp.MeasureFunction.W1,
            hp.NormalizeFunction.NPAIRS,
            group_flatten_method=FlattenGroup.NONE,
            drop_none_labels=False,
            probability_based=True
        )


class AverageIndividualFairnessOnlyClass(
        ProbabilityScoreKeepOnlyClass, CounterfactualPCM):
    def __init__(self):
        ProbabilityScoreKeepOnlyClass.__init__(
            self, use_pos_score_in_binary_sent=False)
        CounterfactualPCM.__init__(
            self,
            "average_individual_fairness_only_target_class",
            hp.MeasureFunction.W1,
            hp.NormalizeFunction.NPAIRS,
            group_flatten_method=FlattenGroup.NONE,
            drop_none_labels=True,
            probability_based=True
        )


# Brokan et al. 2019

class AverageEqualityGap(ProbabilityScoreKeepOnlyClass, GroupBCM):
    def __init__(self):
        ProbabilityScoreKeepOnlyClass.__init__(
            self, use_pos_score_in_binary_sent=True)
        GroupBCM.__init__(
            self,
            "aeg", hp.MeasureFunction.AEG,
            hp.NormalizeFunction.NGROUPS,
            hp.Background.EXCLUDE_GROUP,
            drop_none_labels=True,
            probability_based=True,
            vector_valued=True
        )


class ExtendedAverageEqualityGap(ProbabilityScoreKeepOnlyClass, GroupBCM):
    def __init__(self):
        ProbabilityScoreKeepOnlyClass.__init__(
            self, use_pos_score_in_binary_sent=False)
        GroupBCM.__init__(
            self,
            "extended_aeg", hp.MeasureFunction.AEG,
            hp.NormalizeFunction.NGROUPS,
            hp.Background.EXCLUDE_GROUP,
            drop_none_labels=True,
            probability_based=True,
            vector_valued=True
        )


# Prabhakaran et al. 2019

#  (adapted as PCM to fit templated scenario)

class PerturbationScoreSensitivity(ProbabilityScoreTarget, CounterfactualPCM):
    def __init__(self):
        super().__init__(
            "perturbation_score_sensitivity", hp.MeasureFunction.ABS_DIFF,
            hp.NormalizeFunction.NPAIRS,
            drop_none_labels=True,
            probability_based=True
        )


class PerturbationScoreRange(ProbabilityScoreTarget, CounterfactualMCM):
    def __init__(self):
        CounterfactualMCM.__init__(
            self,
            'perturb_score_range',
            hp.MeasureFunction.RANGE,
            drop_none_labels=True,
            probability_based=True
        )


class PerturbationScoreDeviation(ProbabilityScoreTarget, CounterfactualMCM):
    def __init__(self):
        CounterfactualMCM.__init__(
            self,
            'perturb_score_deviation',
            hp.MeasureFunction.STD,
            drop_none_labels=True,
            probability_based=True
        )


######

class PerturbationScoreRange2(ProbabilityScore, CounterfactualMCM):
    def __init__(self):
        # False is Equivalent to true for the range distance measure
        ProbabilityScore.__init__(self, use_pos_score_in_binary_sent=False)

        CounterfactualMCM.__init__(
            self,
            'perturb_score_range',
            hp.MeasureFunction.RANGE,
            drop_none_labels=False,
            probability_based=True
        )


class PerturbationScoreRangeOnlyClass(
        ProbabilityScoreKeepOnlyClass, CounterfactualMCM):
    def __init__(self):
        # False is Equivalent to true for the range distance measure
        ProbabilityScoreKeepOnlyClass.__init__(
            self, use_pos_score_in_binary_sent=False)

        CounterfactualMCM.__init__(
            self,
            'perturb_score_range_only_target_class',
            hp.MeasureFunction.RANGE,
            drop_none_labels=True,
            probability_based=True
        )


class PerturbationScoreDeviation2(ProbabilityScore, CounterfactualMCM):
    def __init__(self):
        ProbabilityScore.__init__(self, use_pos_score_in_binary_sent=False)

        CounterfactualMCM.__init__(
            self,
            'perturb_score_deviation',
            hp.MeasureFunction.STD,
            drop_none_labels=False,
            probability_based=True
        )


class PerturbationScoreDeviationOnlyClass(
        ProbabilityScoreKeepOnlyClass, CounterfactualMCM):
    def __init__(self):
        ProbabilityScore.__init__(
            self, use_pos_score_in_binary_sent=False)

        CounterfactualMCM.__init__(
            self,
            'perturb_score_deviation_only_target_class',
            hp.MeasureFunction.STD,
            drop_none_labels=False,
            probability_based=True
        )
