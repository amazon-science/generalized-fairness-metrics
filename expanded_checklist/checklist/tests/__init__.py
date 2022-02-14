from .classification import *
from .sequence import *
from .task_agnostic import *


def get_core_tests():
    gmetrics = [
        # prediction based
        FNED(),
        FPED(),

        FixedFNED(),
        FixedFPED(),

        FPRRatioBCM(),
        FNRRatioBCM(),

        TNRGapPCM(),
        TPRGapPCM(),

        DisparsityScore(),
        FixedDisparsityScore(),

        ParityGap(),

    #     FNRRange(),
    #     FPRRange(),

        # probability based:
        AverageGroupFairness(),

        # only class -- analyses the probabilities for class x only for
        # examples with gold label x
        AverageGroupFairnessOnlyClass(),
        AverageEqualityGap(),
        ExtendedAverageEqualityGap()
    ]

    return gmetrics + get_cf_metrics()


def get_cf_metrics():
    return [
        # probability based:
        CounterfactualFairnessGapOnlyClass(),
        AverageIndividualFairnessOnlyClass(),
      #  PerturbationScoreRangeOnlyClass(),
      #  PerturbationScoreDeviationOnlyClass(),
        CounterfactualFairnessGap(),
        AverageIndividualFairness(),
        PerturbationScoreRange(),
        PerturbationScoreDeviation(),
        PerturbationScoreSensitivity()
    ]

def get_all_tests():
    return [INV(), BasicSeqMetrics(), BasicClassificationMetrics()] +\
        get_core_tests()
