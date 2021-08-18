import pandas as pd
import regex as re
from expanded_checklist.checklist.utils import DataShape
from collections import defaultdict

from conversationai.unintended_ml_bias.model_bias_analysis import\
    compute_bias_metrics_for_models

from ..abstract_tests import ClassificationMetric


class SubgroupMetrics(ClassificationMetric):
    def __init__(self) -> None:
        """
        Subgroup, classification threshold agnostic metrics from
        https://arxiv.org/pdf/1903.04561.pdf.
        """
        # only_accumulate set to True -- the get_results function expects to
        # get labels and preds for all groups (not for a single group)
        super().__init__(
            'subgroup_auc',
            required_ds=DataShape.GROUPED,
            only_accumulate=True)

    def get_binary_class_results(self, labels, preds, confs):
        data = defaultdict(list)

        mname = "score"
        for gname, lab, conf in zip(self.group_names, labels, confs):
            data[mname].extend(conf)
            data['label'].extend(lab)
            data['identity'].extend([gname for a in range(len(conf))])
            for gname2 in self.group_names:
                data[gname2].extend(
                    [gname2 == gname for a in range(len(conf))])

        df = pd.DataFrame(data)
        results = compute_bias_metrics_for_models(
            df, self.group_names, [mname], 'label')
        results.columns = [re.sub(f"{mname}_", "", x) for x in results.columns]
        return results

    def summary(self, core_record, res_munch, verbose=False, **kwargs):
        results = res_munch.results
        n_examples = res_munch.n_examples

        print(f"Examples used for evaluation: {n_examples}")
        for cl, res in results.items():
            if not cl:
                continue
            print(f"======== Results for class {cl}")
            print(res)
