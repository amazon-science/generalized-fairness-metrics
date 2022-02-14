
from ..tests.abstract_tests.generalized_metrics import PCM, MCM, BCM
from ..utils import ACCUMULATED_STR, DataShape
from ..tests import get_core_tests

from munch import Munch
from typing import Dict
import os
from enum import Enum

from collections import defaultdict
import numpy as np
import scipy.stats as stats
from sklearn import metrics, preprocessing
import matplotlib.cm as cm
import math

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern Sans serif']})
# rc('text', usetex=True)

import logging
from copy import deepcopy
logger = logging.getLogger(__name__)

test_name2acr = {
    "cfgap": "CFGap",
    "average_individual_fairness": "AvgIF",
    "perturb_score_range": "PertSR",
    "perturb_score_deviation": "PertSD",
    "perturbation_score_sensitivity": "PertSS",
    "fped": "FPED",
    "fned": "FNED",
    "fixed_fped": "FPED*",
    "fixed_fned": "FNED*",
    "fpr_ratio_bcm": "FPR Ratio",
    "fnr_ratio_bcm": "FNR Ratio",
    "aeg": "AvgEG",
    "extended_aeg": "AvgEG",
    "average_group_fairness": "AvgGF",
    "tnr_gap_pcm": "TNR Gap",
    "tpr_gap_pcm": "TPR Gap",
    "fixed_disparsity_score": "Disparity Score*",
    "disparsity_score": "Disparity Score",
    "parity_gap": "Parity Gap"
}

vbcms = {'aeg', 'extended_aeg',
         'fpr_ratio_bcm', 'fnr_ratio_bcm'}


group_metrics_bcms = {
    'fped', 'fned', 'fixed_fped', 'fixed_fned', 'average_group_fairness'
    }

group_metrics_pcms = {
    'tnr_gap_pcm', 'tpr_gap_pcm', 'disparsity_score',
    'fixed_disparsity_score', 'parity_gap'
    }

correct_normalization = {
    'fixed_fped', 'fixed_fned', 'fixed_disparsity_score'
}


class Scaling(Enum):
    NONE = 0
    MAX_ABS = 1
    UNIT_NORM = 2


def get_name(tname, test):
    if "_only_target_class" in tname:
        tname = tname.replace("_only_target_class", "")
        suffix = " (TC)"
    else:
        suffix = ""

    name = test_name2acr[tname] if tname in test_name2acr \
        else tname.replace("_", " ")

    if test.probability_based:
        name = f"{name}$^P$"
    name += suffix
    return name


def plot_all_tests_for_different_cores(
    model_name: str,
    core_name2test2res: Dict,
    out_dir: str,
    cl=1,
    scaling=Scaling.MAX_ABS,
    **kwargs
):
    metric_names = []
    core2scores = defaultdict(list)
    prob_based = []
    counterfactual = []
    metric_type = []
#
    for test in get_core_tests():
        tname = test.get_name()
        if any(vbcm in tname for vbcm in vbcms):
            continue

        if any(tname not in test2res
               for test2res in core_name2test2res.values()) or \
                any(type(test2res[tname]) == Exception
                    for test2res in core_name2test2res.values()):
                #         or \
                # (test.required_ds == DataShape.UNGROUPED
                #     and not counterfactual) or \
                # (test.required_ds == DataShape.GROUPED and counterfactual):
            continue

        name = get_name(tname, test)

        # if "TC" in name and not any(
        #         x in name for x in ["AvgIF", "AvgGF", "CFGap"]):
        #     continue

        metric_names.append(name)
        prob_based.append(test.probability_based)
        counterfactual.append(test.required_ds != DataShape.GROUPED)
        mtype = _get_super_class(test)
        metric_type.append(mtype)

        for core in sorted(core_name2test2res.keys()):
            test2res = core_name2test2res[core]
            res_munch = test2res[tname]

            score = _get_accumulated_score(test, res_munch, cl=cl)
            if score is None:
                score = _get_accumulated_score(test, res_munch, cl="all")
                if score is None:
                    continue
                if "(all)" not in metric_names[-1]:
                    metric_names[-1] = metric_names[-1] + " (all)"
            core2scores[core].append(score)

    group_names = sorted(core2scores.keys())
    core2scores['Metric'] = metric_names
    core2scores['Type'] = metric_type
    core2scores['ProbBased'] = prob_based
    core2scores['Counterfactual'] = counterfactual
    df = pd.DataFrame(core2scores)
    model_and_class = f"{model_name}-{cl}"

    plot_groups(df, group_names,
                f'{model_and_class}-multi-attr-metrics-heatmap.pdf',
                out_dir, last_col=None, tight=True, scaling=scaling)

    # plot_correlation(df, group_names,
    #             f'{model_and_class}-corr.pdf', out_dir, tight=True)
    return df, group_names


def plot_all_tests(
    model_name: str,
    test2res: Dict,
    core_name: str,
    out_dir: str,
    counterfactual: bool,
    cl=1,
    skip_group_pcms: bool = True,
    scaling=Scaling.MAX_ABS,
    **kwargs
):
    """
    For group metrics (if counterfactual is False) plot only the results for
    different groups (and skip pcms if skip_group_pcms).
    """
    metric_names = []
    scores = []
    group2scores = defaultdict(list)
    prob_based = []
    metric_type = []

    indexes_with_blank_group_scores = []

    plot_type = 'cf' if counterfactual else 'group'
    i = 0
    for test in get_core_tests():
        tname = test.get_name()

        print(tname)
        # skip normalised versions (they don't affect group results)
        if (skip_group_pcms and any(x in tname for x in group_metrics_pcms)) \
                or tname in correct_normalization:
            continue

        if (tname not in test2res) or \
                (test.required_ds == DataShape.UNGROUPED
                 and not counterfactual) or \
                (test.required_ds == DataShape.GROUPED and counterfactual):
            continue

        try:

            res_munch = test2res[tname]
            # missing res
            if type(res_munch) == Exception:
                logger.warning(f"Missing results for test {tname}.")
                logger.exception(f"Reason: {res_munch}")
                continue

            name = get_name(tname, test)

            if "TC" in name and not any(
                    x in name for x in ["AvgIF", "AvgGF", "CFGap"]):
                continue

            score = _get_accumulated_score(test, res_munch, cl=cl)
            if score is None:
                score = _get_accumulated_score(test, res_munch, cl="all")
                if score is None:
                    continue
                name += " (all)"

            mtype = _get_super_class(test)

            if test.required_ds == DataShape.GROUPED:
                # this appends the scores to apprropriate lists in group2scores
                _get_group_scores(test, group2scores, res_munch, cl=cl)
            else:
                indexes_with_blank_group_scores.append(i)

            scores.append(score)
            metric_names.append(name)
            prob_based.append(test.probability_based)
            metric_type.append(mtype)

            # binary classification; aeg should be calculated for positive and
            # for neg classes (if class was not binary aeg would not have been
            # computed)
            if tname in ["aeg", "average_group_fairness_only_target_class"]: \
               #     and cl == 1:
                metric_names[-1] = f"Pos{metric_names[-1]}"
                score = _get_accumulated_score(test, res_munch, cl=0)
                _get_group_scores(test, group2scores, res_munch, cl=0)
                scores.append(score)
                metric_names.append(f"Neg{name}")
                prob_based.append(test.probability_based)
                metric_type.append(mtype)
                i += 1
            i += 1

        except Exception as err:
            logger.warning(f"Missing results for the test. Reason: {err}")

    for _, gscores in group2scores.items():
        for i in indexes_with_blank_group_scores:
            gscores.insert(i, None)
    # # don't show accumulated in the per-group split
    # if 'accumulated' in group2scores:
    #     del group2scores['accumulated']

    if group2scores:
        group_names = sorted(group2scores.keys())
    else:
        group_names = None

    group2scores['Accumulated'] = scores
    group2scores['Metric'] = metric_names
    group2scores['Type'] = metric_type
    group2scores['ProbBased'] = prob_based
    group2scores['Score'] = scores

    df = pd.DataFrame(group2scores)

    model_and_class = f"{model_name}-{cl}"
    plot_groups(
        df, group_names,
        f'{model_and_class}-{core_name}-{plot_type}-metrics-heatmap.pdf',
        out_dir,
        scaling=scaling)

    return df, group_names


###############################
#   CORE PLOTTING FUNCITONS   #
###############################

# each column is a different metric, those that are only meant to be
# used for scaling should be marked with "DROP" string in the name
def plot_groups(
        df, group_names, plot_name, out_dir,
        scores=None, last_col="accumulated", tight=False,
        scaling=Scaling.NONE):
    try:
        df = df.set_index('Metric')
    except:
        pass

    if 'PosAvgEG$^P$' in df.index:
        df = df.drop(['AvgEG$^P$'])

    bcm_metrics = df[df['Type'] == "BCM"]
    pcm_metrics = df[df['Type'] == "PCM"]
    mcm_metrics = df[df['Type'] == "MCM"]

    df = df.sort_values(
        by=['ProbBased'], ascending=True)

    df = pd.concat([bcm_metrics,
                    pcm_metrics,
                    mcm_metrics])
    df.round(3)

    metric_counts = [[len(bcm_metrics), len(pcm_metrics), len(mcm_metrics)]]
    if group_names:
        columns = group_names
        if last_col and last_col in columns:
            columns = [c for c in columns if c != last_col] + [last_col]

        clean_columns = [x for x in columns if "DROP" not in x]
        num_columns = len(clean_columns)
        num_rows = len(df)
        xticklabels = [c.replace("_", " ") for c in clean_columns]

        yticklabels = "auto"
        if 'Counterfactual' in df:
            dfs = [
                df[df['Counterfactual'] == True][columns],
                df[df['Counterfactual'] == False][columns]
            ]
            metric_counts = [
                [
                    len(bcm_metrics[bcm_metrics['Counterfactual'] == True]),
                    len(pcm_metrics[pcm_metrics['Counterfactual'] == True]),
                    len(mcm_metrics[mcm_metrics['Counterfactual'] == True])
                ],
                [
                    len(bcm_metrics[bcm_metrics['Counterfactual'] == False]),
                    len(pcm_metrics[pcm_metrics['Counterfactual'] == False]),
                    len(mcm_metrics[mcm_metrics['Counterfactual'] == False])
                ]
            ]
        else:
            dfs = [df[columns]]

    else:
        # 1D heatmap: no group results = counterfactual metrics
        num_columns = len(df)
        num_rows = 1
        data = [list(df['Score'])]
        xticklabels = list(df.index)
        yticklabels = []
        dfs = [data]

    fig, axs = plt.subplots(
        len(dfs), 1, sharex=False, sharey=False,
        figsize=(0.8 * num_columns, 0.5 * num_rows),
        gridspec_kw={'height_ratios': [len(x) for x in dfs]})

    if type(axs) not in [list, np.ndarray]:
        axs = [axs]

    for i, (data, mcounts) in enumerate(zip(dfs, metric_counts)):
        if i != 0:
            xticks = []
        else:
            xticks = xticklabels

        if type(scaling) != Scaling:
            try:
                scaling = Scaling(scaling)
            except:
                scaling = Scaling.NONE

        cbar = False
        vmin, vmax = -1, 1

        accumulated_cols = None
        if type(data) == list:
            # no heatmap
            scaled_data = [[0] * len(data[0])]
        else:
            if scaling == Scaling.NONE:
                scaled_data = data
                vmin, vmax = -0.2, 0.2
                cbar = True
            else:
                # round to have clean results with scaling if all measurements are
                # very close to 0
                data = data.round(3)

                accumulated_cols = [x for x in data.columns if 'accumulated' in x]
                tmp_data = data.drop(columns=accumulated_cols)

                if scaling == Scaling.UNIT_NORM:
                    x = tmp_data.values
                    x_scaled = preprocessing.normalize(x, norm='l2')
                    scaled_data = pd.DataFrame(x_scaled)
                elif scaling == Scaling.MAX_ABS:
                    x = tmp_data.values.T
                    scaler = preprocessing.MaxAbsScaler()
                    x_scaled = scaler.fit_transform(x)
                    scaled_data = pd.DataFrame(x_scaled)
                    scaled_data = scaled_data.T
                scaled_data.index = tmp_data.index
                scaled_data.columns = tmp_data.columns

                for ac in accumulated_cols:
                    scaled_data[ac] = [0 if not math.isnan(x) else None for x in data[ac]]

            scaled_data = scaled_data[scaled_data.columns.drop(list(scaled_data.filter(regex='DROP')))]
            data = data[data.columns.drop(list(data.filter(regex='DROP')))]

        sns.heatmap(
            scaled_data,
            xticklabels=xticks,
            yticklabels=yticklabels,
            annot=data,
            fmt='.3f',
            cbar=cbar,
            cmap=sns.diverging_palette(220, 20, 55 ,l=60, as_cmap=True),
            #sns.color_palette("Reds", as_cmap=True),
            vmax=vmax,
            vmin=vmin,
            ax=axs[i]
        )
        axs[i].tick_params('x', labelrotation=20)
        #axs[i].tick_params('y', labelrotation=30)

        axs[i].xaxis.tick_top()
        axs[i].yaxis.set_label_text('')

        if not group_names:
            axs[i].vlines(
                [mcounts[0], mcounts[0] + mcounts[1]],
                *axs[i].get_ylim(),
                colors='black', linestyles='dashed')
        elif last_col and last_col in columns:
            axs[i].vlines(
                [num_columns - 1], *axs[i].get_ylim(),
                colors='black', linestyles='dashed')

        axs[i].hlines(
            [mcounts[0], mcounts[0] + mcounts[1]],
            *axs[i].get_xlim(),
            colors='black', linestyles='dashed')

    fname = f'{out_dir}/{plot_name}'
    # plt.xticks(rotation=20)

    if tight:
        plt.tight_layout()
    plt.savefig(fname, bbox_inches='tight', pad_inches=0)
    plt.show()


def plot_correlation(
        df, group_names, plot_name, out_dir,
        scores=None, tight=False,
        scaling=Scaling.NONE):
    try:
        df = df.set_index('Metric')
    except:
        pass

    if 'PosAvgEG$^P$' in df.index:
        df = df.drop(['AvgEG$^P$'])

    mcounts = len(df)
    assert group_names

    columns = group_names
    num_columns = len(columns)
    num_rows = len(df)
    yticklabels = "auto"

    df = df[columns]

    fig = plt.figure(figsize=(0.5 * num_rows,  0.3 * num_rows))
    ax = fig.add_subplot(111)
    corrMatrix = df.T.corr('kendall')

    sns.heatmap(
        corrMatrix,
        xticklabels=df.index,
        yticklabels=df.index,
        annot=False,
        fmt='.3f',
        cbar=True,
        #cmap=sns.diverging_palette(220, 20, 55 ,l=60, as_cmap=True),
        #sns.color_palette("Reds", as_cmap=True),
        vmax=1,
        vmin=-1
    )
    ax.tick_params('x', labelrotation=45)
    #axs[i].tick_params('y', labelrotation=30)

    ax.xaxis.tick_top()
    ax.yaxis.set_label_text('')

    fname = f'{out_dir}/{plot_name}'
    # plt.xticks(rotation=20)

    if tight:
        plt.tight_layout()
    plt.savefig(fname, bbox_inches='tight', pad_inches=0)
    plt.show()



def plot_barplot(df, plot_name, out_dir):
    pcm_metrics = df[df['Type'] == "PCM"]
    mcm_metrics = df[df['Type'] == "MCM"]
    bcm_metrics = df[df['Type'] == "BCM"]

    sp_contents = [
        ("PCM Metrics", pcm_metrics), ("MCM Metrics", mcm_metrics),
        ("BCM Metrics", bcm_metrics)
    ]

    sp_contents = [(x, y) for x, y in sp_contents if len(y) > 0]

    fig, axs = plt.subplots(
        1, len(sp_contents), sharex=False, sharey=True,
        figsize=(len(df), 4),
        gridspec_kw={'width_ratios': [len(y) for x, y in sp_contents]})

    for i, (title, data) in enumerate(sp_contents):
        axs[i].set_title(title)
        sns.barplot(
            x='Metric', y='Score', data=data, ax=axs[i], palette="Blues_d")
        axs[i].set_xlabel('')
        if i != 0:
            axs[i].set_ylabel('')
        axs[i].tick_params('x', labelrotation=20)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    fname = f'{out_dir}/{plot_name}'
    plt.tight_layout()
    plt.savefig(fname)
    plt.show()


def per_subgroup_scatterplots(
        group_names, values,
        title='', y_lim=(0.1, 1.0), figsize=(15, 5),
        point_size=20, file_name='plot'):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    for i, val in enumerate(values):
        x = [i] * len(val)
        y = val
        ax.scatter(x, y, s=point_size)

    ax.set_xticklabels(group_names, rotation=90)
    ax.set_xticks(list(range(len(group_names))))
    ax.set_ylim(y_lim)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(f'/tmp/{file_name}.eps', format='eps')


###############
#   HELPERS   #
###############

def _get_accumulated_score(test, res_munch, cl):
    results = res_munch.results
    if cl in results:
        name2score = results[cl]
        if ACCUMULATED_STR in name2score:
            return name2score[ACCUMULATED_STR]
    return None


def _get_group_scores(test, group2scores, res_munch, cl):
    results = res_munch.results
    if cl in results:
        name2score = results[cl]
        for name, score in name2score.items():
            if type(name) == tuple:
                # skip tuples of group names (PCM)
                continue

            # the metric is not meant to be accumulated
            if name == ACCUMULATED_STR and test.get_name() in vbcms:
                score = None
            name = str(name)
            group2scores[name].append(score)


def _get_super_class(test):
    all_classes = test.__class__.mro()
    all_classes = [str(x) for x in all_classes]

    if str(PCM) in all_classes:
        return "PCM"
    elif str(MCM) in all_classes:
        return "MCM"
    elif str(BCM) in all_classes:
        return "BCM"
    else:
        return "?"
