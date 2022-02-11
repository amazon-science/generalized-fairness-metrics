from enum import Enum
from typing import Callable, Tuple, Optional
import itertools
from scipy import stats
from functools import partial

import numpy as np
from ..utils import is_1d_list, is_2d_list
from sklearn.metrics import jaccard_score

import logging
logger = logging.getLogger(__name__)


####################################
#   HELPER FUNS & CLASSES FOR      #
#   QUICK METRIC CREATION          #
####################################


class Background(Enum):
    ALL_GROUPS = 1
    EXCLUDE_GROUP = 2


def get_data_for_many_groups(
        orig_labels, orig_preds, orig_confs, orig_meta, gindex_to_skip=None):
    new_labels, new_preds, new_confs, new_meta = [], [], [], []

    for gindex, (labels, preds, confs, meta) in enumerate(zip(
            orig_labels, orig_preds, orig_confs, orig_meta)):
        if gindex_to_skip is not None and gindex == gindex_to_skip:
            continue

        new_labels += [x for x in labels]
        new_preds += [x for x in preds]
        new_confs += [x for x in confs]
        new_meta += [x for x in meta]
    return new_labels, new_preds, new_confs, new_meta


def get_bg_all(labels, preds, confs, meta, gindex, gname) -> Tuple:
    return get_data_for_many_groups(
        labels, preds, confs, meta, gindex_to_skip=None)


def get_bg_all_except(labels, preds, confs, meta, gindex, gname) -> Tuple:
    # background are all groups apart from the given one
    return get_data_for_many_groups(
        labels, preds, confs, meta, gindex_to_skip=gindex)


def get_background_fun(bg: Background) -> Callable:
    if bg == Background.ALL_GROUPS:
        return get_bg_all
    elif bg == Background.EXCLUDE_GROUP:
        return get_bg_all_except
    else:
        raise Exception(f'Not supported background type: {bg}')


####################

class MeasureFunction(Enum):
    ABS_DIFF = 1
    RATIO_LEFT = 2
    RATIO_RIGHT = 3
    AEG = 4
    W1 = 5
    RANGE = 6
    STD = 7
    JACCARD_DIST = 8


def get_scalar_measure_funs():
    return [
        MeasureFunction.ABS_DIFF,
        MeasureFunction.RATIO_LEFT,
        MeasureFunction.RATIO_RIGHT]


def aeg_fun(score1, score2) -> Optional[int]:
    assert is_1d_list(score1) and is_1d_list(score2)
    n1 = len(score1)
    n2 = len(score2)
    if n1 == 0 or n2 == 0:
        return None

    # first score is the background so MWU stat is greater
    u, _ = stats.mannwhitneyu(score1, score2, alternative='greater')
    return 0.5 - u / (n1 * n2)


def wasserstein_dist_1(score1, score2) -> Optional[int]:
    assert is_1d_list(score1) and is_1d_list(score2)
    return stats.wasserstein_distance(score1, score2)


def abs_diff(score1, score2) -> Optional[int]:
    if np.isscalar(score1) and np.isscalar(score2):
        return abs(score1 - score2)
    else:
        # this can happend for counterfactual settings based on probs, with seq
        assert is_1d_list(score1) and is_1d_list(score2) and \
            len(score1) == len(score2)
        tmp = [abs_diff(x, y) for x, y in zip(score1, score2)]
        return sum(tmp)/(len(tmp) + 1e-13)


def ratio(score1, score2, left=bool) -> Optional[int]:
    if np.isscalar(score1) and np.isscalar(score2):
        if left:
            return score1/(score2 + 1e-13)
        else:
            return score2/(score1 + 1e-13)
    else:
        # handling counterfactual seq labeling
        assert is_1d_list(score1) and is_1d_list(score2) and \
            len(score1) == len(score2)
        logger.warning("Returning avergage ratio over tokens. Is this what \
            you intended?")
        tmp = [ratio(x, y, left) for x, y in zip(score1, score2)]
        return sum(tmp)/(len(tmp) + 1e-13)


def range_fun(scores) -> Optional[int]:
    if is_1d_list(scores):
        return max(scores) - min(scores)
    else:
        # handling counterfactual seq labeling
        assert is_2d_list(scores)
        tmp = [range_fun(list(x)) for x in zip(*scores)]
        return sum(tmp)/(len(tmp) + 1e-13)


def std_fun(scores) -> Optional[int]:
    if is_1d_list(scores):
        return np.std(scores)
    else:
        # handling counterfactual seq labeling
        assert is_2d_list(scores)
        tmp = [std_fun(list(x)) for x in zip(*scores)]
        return sum(tmp)/(len(tmp) + 1e-13)


def jaccard_dist_fun(score1, score2) -> Optional[int]:
    return 1 - jaccard_score(score1, score2)


def get_measure_fun(mf: MeasureFunction) -> Callable:
    if mf == MeasureFunction.ABS_DIFF:
        return abs_diff
    elif mf == MeasureFunction.RATIO_LEFT:
        return partial(ratio, left=True)
    elif mf == MeasureFunction.RATIO_RIGHT:
        return partial(ratio, left=False)
    elif mf == MeasureFunction.AEG:
        return aeg_fun
    elif mf == MeasureFunction.W1:
        return wasserstein_dist_1
    elif mf == MeasureFunction.RANGE:
        return range_fun
    elif mf == MeasureFunction.STD:
        return std_fun
    elif mf == MeasureFunction.JACCARD_DIST:
        return jaccard_dist_fun
    else:
        raise Exception(f'Not supported measure function type: {mf}')


####################

class NormalizeFunction(Enum):
    NONE = 1
    NGROUPS = 2
    NPAIRS = 3


def do_not_normalize(full_gap: int, n_groups: int) -> int:
    return full_gap


def normalize_ngroups(full_gap: int, n_groups: int) -> int:
    return full_gap/n_groups


def normalize_npairs(full_gap: int, n_groups: int) -> int:
    return full_gap/len(list(itertools.combinations(range(n_groups), 2)))


def get_normalize_fun(nf: NormalizeFunction) -> Callable:
    if nf == NormalizeFunction.NONE:
        return do_not_normalize
    elif nf == NormalizeFunction.NGROUPS:
        return normalize_ngroups
    elif nf == NormalizeFunction.NPAIRS:
        return normalize_npairs
    else:
        raise Exception(f'Not supported normalization function type: {nf}')
