from scipy import stats

from permute.core import one_sample, two_sample


def get_exception(test_name):
    return Exception(f"Can't run a {test_name}. Incorrect data. " +
                     "Only binary classification supported atm")


def check_paired_sample_format(data):
    return len(data) == 2 and all(len(x) == 2 for x in data[0]) and \
        all(len(x) == 2 for x in data[1])


def paired_ttest(confs):
    # run paired sample t-test only on binary prediction models
    if not check_paired_sample_format(confs):
        raise get_exception("paired sample t-test")

    scores1 = [x[1] for x in confs[0]]
    scores2 = [x[1] for x in confs[1]]
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


def wilcoxon(confs):
    if not check_paired_sample_format(confs):
        raise get_exception("wilcoxon test")

    scores1 = [x[1] for x in confs[0]]
    scores2 = [x[1] for x in confs[1]]
    diffs = [x1 - x2 for x1, x2 in zip(scores1, scores2)]
    stat, pval = stats.wilcoxon(diffs)
    info = ""
    return stat, pval, info


def permutation_one_sample(confs, stat='t'):
    if not check_paired_sample_format(confs):
        raise get_exception("paired permutation test")

    scores1 = [x[1] for x in confs[0]]
    scores2 = [x[1] for x in confs[1]]

    # print(scores1[:20])
    # print(scores2[:20])

    pval, diff_means = one_sample(
        scores1, scores2, stat="mean", alternative='two-sided')
    info = ""
    return diff_means, pval, info


def permutation_two_sample(confs, stat='t'):
    if not check_paired_sample_format(confs):
        raise get_exception("paired permutation test")

    scores1 = [x[1] for x in confs[0]]
    scores2 = [x[1] for x in confs[1]]

    # print(scores1[:20])
    # print(scores2[:20])

    pval, diff_means = two_sample(
        scores1, scores2, stat="mean", alternative='two-sided')
    info = ""
    return diff_means, pval, info
