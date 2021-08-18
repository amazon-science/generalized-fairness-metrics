from scipy import stats

from permute.core import one_sample, two_sample
import numpy as np


def get_exception(test_name):
    return Exception(f"Can't run a {test_name}. Incorrect data. " +
                     "Only two groups supported.")


def paired_ttest(confs, idx=None):
    # run paired sample t-test only on binary prediction models
    if len(confs) != 2:
        raise get_exception("paired sample t-test")

    scores1 = [x[idx] for x in confs[0]]
    scores2 = [x[idx] for x in confs[1]]
    _, pval1 = stats.shapiro(scores1)
    _, pval2 = stats.shapiro(scores2)

    if pval1 < 0.05 or pval2 < 0.05:
        info =\
            "WARNING: Assumptions are likely violated. Data is unlikely to " +\
            "be normally distributed."
        info += f"\nP-values from Shapiro-Wilk: {pval1:.4f}, {pval2:.4f}"
    else:
        info =\
            "Assumptions are likely to hold. Data is likely to " +\
            "be normally distributed."
        info += f"\nP-values from Shapiro-Wilk: {pval1:.4f}, {pval2:.4f}"

    stat, pval = stats.ttest_rel(scores1, scores2)
    return stat, pval, info


def wilcoxon(confs, idx=None):
    if len(confs) != 2:
        raise get_exception("wilcoxon test")

    scores1 = [x[idx] for x in confs[0]]
    scores2 = [x[idx] for x in confs[1]]

    diffs = [x1 - x2 for x1, x2 in zip(scores1, scores2)]
    stat, pval = stats.wilcoxon(diffs)
    info = ""
    return stat, pval, info


def permutation_one_sample(confs, stat='t', idx=None):
    if len(confs) != 2:
        raise get_exception("paired permutation test")

    scores1 = [x[idx] for x in confs[0]]
    scores2 = [x[idx] for x in confs[1]]

    pval, diff_means = one_sample(
        scores1, scores2, stat="mean", alternative='two-sided', reps=1000)
    info = ""
    return diff_means, pval, info


def permutation_two_sample(confs, stat='t', idx=None):
    if len(confs) != 2:
        raise get_exception("permutation test")

    scores1 = [x[idx] for x in confs[0]]
    scores2 = [x[idx] for x in confs[1]]

    pval, diff_means = two_sample(
        scores1, scores2, stat="mean", alternative='two-sided', reps=1000)
    info = ""
    return diff_means, pval, info


def mann_whitney_u(confs, idx=None):
    if len(confs) != 2:
        raise get_exception("mann whitney u")

    scores1 = [x[idx] for x in confs[0]]
    scores2 = [x[idx] for x in confs[1]]
    stat, pval = stats.mannwhitneyu(scores1, scores2, alternative='two-sided')
    info = ""
    return stat, pval, info


def kruskal(confs, idx=None):
    scores = []
    for conf in confs:
        scores.append([x[idx] for x in conf])

    stat, pval = stats.kruskal(*scores)
    info = ""
    return stat, pval, info


def friedman(confs, idx=None):
    scores = []
    for conf in confs:
        scores.append([x[idx] for x in conf])

    stat, pval = stats.friedmanchisquare(*scores)
    info = ""
    return stat, pval, info
