from expanded_checklist.checklist.utils import DataShape, \
    convert_class_data_to_seq_data, get_class_from_seq_label
from expanded_checklist.checklist.tests.abstract_tests import MetricTest
from itertools import combinations
from collections import defaultdict
from tabulate import tabulate


class CounterfactualGap(MetricTest):
    def __init__(self, condition: str = 'all', n_classes: int = None) -> None:
        """
        Counterfactual Fairness Gap from https://arxiv.org/pdf/1809.10610.pdf.
        One alteration has been applied: since in the templated sentence
        instantiation there is no 'original' example, we considee the average
        gap in prediction over all possible instantiation pairs.

        Arguments:
            condition: condition determines how the difference in probabilities
            are computed. 4 options are supported:
                1) 'all'
                  for each class (ORG, LOC etc) calculate the differences
                  across *all* tokens (e.g. differences for ORG probabilities
                  will also be calculated on tokens which were of class LOC or
                  O)
                2) 'entity'
                  for each class (ORG, LOC etc) calculate the differences
                  across tokens that have entity gold label (e.g. differences
                  for ORG probabilities will also be calculated on tokens which
                  were of class LOC *but not* for those of class O)
                3) 'gold'
                  for each class (ORG, LOC etc) calculate the differences
                  across tokens that have gold label with that class (e.g.
                  differences for ORG probabilities will only be calculated on
                  tokens which have gold label ORG)
                4) 'matrix'
                  for each class c (ORG, LOC etc) calculate n different scores,
                  where n is the number of classes. Each score is computed by
                  calculating differences for probability of class c, when
                  the gold label is class c2. This is done for all possible c2.
                  This produces a matrix of scores.
            The last 3 conditions can only be used if gold labels are provided.

            TODO: Adjust to the case when the model only provides confidence
            scores for predicted label
        """
        if condition not in ['all', 'entity', 'gold', 'matrix']:
            raise Exception('Unsupported condition!')

        self.condition = condition
        super().__init__('counterfactual_fairness_gap' + "_" + condition,
                         required_ds=DataShape.UNGROUPED,
                         only_accumulate=True,
                         probability_based=True,
                         drop_none_labels=False)

    def get_results(self, labels, preds, confs, meta, **kwargs):
        if not self.seq:
            labels, preds, confs = convert_class_data_to_seq_data(
                labels, preds, confs, self.required_ds)

        n_groups = len(self.group_names)
        possible_pairs = list(combinations(list(range(n_groups)), 2))

        cond = self.condition
        class_results = defaultdict(dict) if cond == 'matrix' else {}

        # consider each two classes
        for cl_idx, cl in enumerate(self.classes):
            for cl2_idx, cl2 in enumerate(self.classes):
                if cond != 'matrix' and cl_idx != cl2_idx:
                    # this inner loop is only used in the matrix condition,
                    # if cond is not matrix, pass through the code only once
                    continue

                ctf_scores = []
                total_tokens_included = 0
                for sent_confs, sent_labels in zip(confs, labels):
                    example_ctf = 0
                    tokens_included = 0
                    n_tokens = len(sent_confs[0])

                    for t in range(n_tokens):
                        if sent_labels is None:
                            tlab = None
                        else:
                            tlab = sent_labels[t]
                            tlab = tlab if not self.seq \
                                else get_class_from_seq_label(tlab)

                        if cond == "all" or \
                                (cond == "entity" and tlab != "O") or \
                                (cond == "gold" and tlab == cl) or \
                                (cond == "matrix" and tlab == cl2):

                            tokens_included += 1
                            for gidx1, gidx2 in possible_pairs:
                                score1 = sent_confs[gidx1][t][cl_idx]
                                score2 = sent_confs[gidx2][t][cl_idx]
                                example_ctf += abs(score1 - score2)

                    # exclude the example if there were no relevant tokens
                    if tokens_included != 0:
                        total_tokens_included += tokens_included
                        example_ctf = example_ctf/tokens_included
                        example_ctf = example_ctf/len(possible_pairs)
                        ctf_scores.append(example_ctf)

                CTF_gap =\
                    sum(ctf_scores)/len(ctf_scores) if ctf_scores else None
                if cond == 'matrix':
                    class_results[cl][cl2] = (CTF_gap, total_tokens_included)
                else:
                    class_results[cl] = (CTF_gap, total_tokens_included)
        return class_results

    def summary(self, core_record, res_munch, verbose=False, **kwargs):
        results = res_munch.results
        n_examples = res_munch.n_examples

        print(f"Examples used for evaluation: {n_examples}")
        print(f"Condition: {self.condition}\n")

        if self.condition == "matrix":
            header = ["CLASS on which calculated->"] + self.classes
            total_tokens_row = ["#Tokens"]
            rows = []
            for cl, cl2res in results.items():
                row = [cl]
                for i, cl2 in enumerate(self.classes):
                    score, total_tokens = cl2res[cl2]
                    if type(score) == str:
                        row.append(f"{score}")
                    elif score is None:
                        row.append("x")
                    else:
                        row.append(f"{score:.4f}")
                    if len(total_tokens_row) == i + 1:
                        total_tokens_row.append(total_tokens)
                    else:
                        assert total_tokens_row[i + 1] == total_tokens
                rows.append(row)
            print(tabulate([header] + [total_tokens_row] + rows))
        else:
            header = ["CLASS:"]
            scores = ["CFT Gap:"]
            tokens = ["#Tokens:"]
            for cl, (CFT_gap, total_tokens) in results.items():
                header.append(cl)
                if type(CFT_gap) == str:
                    scores.append(CFT_gap)
                elif CFT_gap is None:
                    scores.append("x")
                else:
                    scores.append(f"{CFT_gap:.4f}")
                tokens.append(total_tokens)
            print(tabulate([header, scores]))
