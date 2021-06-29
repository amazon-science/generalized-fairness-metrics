from tabulate import tabulate
from collections import defaultdict
from itertools import combinations
import numpy as np

from ..eval_core import CoreRecord
from .metric_test import MetricTest
from ..utils import accumulate_ner_probs, DataShape, \
    average_conf_across_tokens_for_identity_term, \
    convert_class_data_to_seq_data


class ProbDiffStats(MetricTest):
    def __init__(self, condition: str, seq: bool = True):
        """
        For possible settings of condition see SeqCounterfactualGap
        """
        if condition not in ['all', 'entity', 'gold', 'matrix']:
            raise Exception('Unsupported condition!')

        self.condition = condition
        super().__init__('probability_diff_stats' + "_" + condition,
                         required_ds=DataShape.UNGROUPED,
                         only_accumulate=True)

    def process_labels_preds_and_confs(
        self,
        core_record: CoreRecord
    ) -> None:
        # ensure that the data is in the ungrouped format
        super().process_labels_preds_and_confs(core_record)

        if core_record.sequence:
            # TODO: this is NER specific - the code should be restructured
            # when more tasks are added
            # accumulate the probabilities for each class
            # e.g. sum the probabilities for B-ORG, I-ORG, L-ORG
            accumulate_ner_probs(core_record)

            average_conf_across_tokens_for_identity_term(core_record)
        else:
            convert_class_data_to_seq_data(core_record)
        self.new_softmax_vocab = core_record.label_vocab
        self.seq = core_record.sequence

    def get_results(self, labels, preds, confs, meta, **kwargs):
        n_groups = len(self.group_names)
        possible_pairs = list(combinations(list(range(n_groups)), 2))

        cond = self.condition

        # this can be either groups2cl2diffs or
        # groups2cl2cl2diffs -- if matrix is the condition
        groups2dict = defaultdict(dict)

        for cl_idx, cl in enumerate(self.new_softmax_vocab):
            for cl2_idx, cl2 in enumerate(self.new_softmax_vocab):
                if cond != 'matrix' and cl_idx != cl2_idx:
                    # this inner loop is only used in the matrix condition,
                    # if cond is not matrix, pass through the code once
                    continue

                for sent_confs, sent_labels in zip(confs, labels):
                    n_tokens = len(sent_confs[0])

                    for t in range(n_tokens):
                        if sent_labels is None:
                            tlab = None
                        else:
                            tlab = sent_labels[t]
                            tlab =\
                                tlab if not self.seq or tlab == "O" \
                                else tlab[2:]

                        if cond == "all" or \
                                (cond == "entity" and tlab != "O") or \
                                (cond == "gold" and tlab == cl) or \
                                (cond == "matrix" and tlab == cl2):

                            for gidx1, gidx2 in possible_pairs:
                                g1 = self.group_names[gidx1]
                                g2 = self.group_names[gidx2]
                                score1 = sent_confs[gidx1][t][cl_idx]
                                score2 = sent_confs[gidx2][t][cl_idx]
                                conf_diff = score1 - score2
                                if cond == 'matrix':
                                    cl2cl2list = groups2dict[(g1, g2)]
                                    cl2list = cl2cl2list.setdefault(cl, {})
                                    lst = cl2list.setdefault(cl2, [])
                                    lst.append(conf_diff)
                                else:
                                    cl2list = groups2dict[(g1, g2)]
                                    lst = cl2list.setdefault(cl, [])
                                    lst.append(conf_diff)
        return groups2dict

    def print_non_matrix(self, groups2dict):
        for (g1, g2), cl2diff_list in groups2dict.items():
            print(f"\n========\nCondition: {g1} vs {g2}")

            header = ["CLASS:"]
            scores1 = [f"{g1} > {g2} %"]
            scores2 = [f"{g1} < {g2} %"]
            mean_row = [f"mean"]
            std_row = [f"std"]
            ntoks_row = [f"#tokens"]

            for cl, diff_list in cl2diff_list.items():
                total = len(diff_list)

                if total != 0:
                    g1_greater = len([x for x in diff_list if x > 0])/total
                    g2_greater = len([x for x in diff_list if x < 0])/total
                    # rest = 1 - g1_greater - g2_greater
                    mean = f"{np.mean(diff_list):.3f}"
                    std = f"{np.std(diff_list):.3f}"
                    ntoks = total
                    res1 = f"{g1_greater*100:.1f}%"
                    res2 = f"{g2_greater*100:.1f}%"
                else:
                    res1 = res2 = mean = std = ntoks = "x"
                header.append(cl)
                scores1.append(res1)
                scores2.append(res2)
                mean_row.append(mean)
                std_row.append(std)
                ntoks_row.append(ntoks)

            print(tabulate(
                [header, scores1, scores2, mean_row, std_row, ntoks_row]))

    def print_matrix(self, groups2dict):
        for (g1, g2), cl2cl2diff_list in groups2dict.items():
            print(f"\n========\nCondition: {g1} > {g2} (%, avg, std, #toks)")

            header = ["CLASS on which calculated->"] + self.new_softmax_vocab
            rows = []

            for cl, cl2diff_list in cl2cl2diff_list.items():
                row = [cl]
                for _, cl2 in enumerate(self.new_softmax_vocab):
                    try:
                        cl2 = int(cl2)
                    except Exception:
                        pass

                    if cl2 not in cl2diff_list:
                        row.append("x")
                        continue

                    diff_list = cl2diff_list[cl2]
                    total = len(diff_list)

                    if total != 0:
                        g1_greater = len([x for x in diff_list if x > 0])/total
                        mean = f"{np.mean(diff_list):.3f}"
                        std = f"{np.std(diff_list):.3f}"
                        ntoks = total
                        res = f"{g1_greater*100:.1f}%, {mean}, {std}, {ntoks}"
                    else:
                        res = "x"

                    row.append(res)
                rows.append(row)
            print(tabulate([header] + rows))

    def summary(self, core_record, res_munch, **kwargs) -> None:
        groups2dict = res_munch.results

        if self.condition == "matrix":
            self.print_matrix(groups2dict)
        else:
            self.print_non_matrix(groups2dict)
