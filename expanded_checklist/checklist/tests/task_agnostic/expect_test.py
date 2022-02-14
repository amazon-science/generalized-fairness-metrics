from .expect import Expect
# from .viewer.test_summarizer import TestSummarizer
from expanded_checklist.checklist.utils import iter_with_optional, DataShape, \
    FlattenGroup, is_1d_list

import numpy as np
from munch import Munch
from ..abstract_tests import AbstractTest

from expanded_checklist.checklist.core_record import CoreRecord
from overrides import overrides


###########################
#       BASE CLASS        #
###########################

class ExpectTest(AbstractTest):
    def __init__(self, expect, agg_fn='all', print_first=None, name=None):
        self.expect = expect
        self.agg_fn = agg_fn
        self.print_first = print_first
        self._name = name
        self._required_ds = DataShape.UNGROUPED
        self._probability_based = False
        self._drop_none_labels = False
        self._group_flatten_method = FlattenGroup.RANDOM_MATCH

    def compute(self, core_record: CoreRecord) -> Munch:
        if core_record.data_structure != self._required_ds:
            raise Exception(
                f'Wrong data structure {core_record.data_structure}')
            # raise Exception('Cannot run a test for ungrouped/counterfatual ' +
            #                 'data on data split into groups!')

        core_record.str_preds = core_record.preds
        results = self.expect(core_record)
        passed = Expect.aggregate(results, self.agg_fn)
        return Munch({"results": results, "passed": passed})

# ====================  SUMMARY

    def _extract_examples_per_testcase(
        self, xs, preds, confs, expect_results,
        labels, meta, nsamples, only_include_fail=True
    ):
        iters = list(iter_with_optional(xs, preds, confs, labels, meta))
        idxs = [0] if self.print_first else []

        idxs = [i for i in np.argsort(expect_results) if
                not only_include_fail or expect_results[i] <= 0]

        if preds is None or (type(preds) == list and len(preds) == 0) or \
                len(idxs) > len(iters):
            return None
        if self.print_first:
            if 0 in idxs:
                idxs.remove(0)
            idxs.insert(0, 0)
        idxs = idxs[:nsamples]
        iters = [iters[i] for i in idxs]
        return idxs, iters, [expect_results[i] for i in idxs]

    def _print(
        self, xs, preds, confs, expect_results,
        labels=None, meta=None, format_example_fn=None, nsamples=3
    ):
        result = self._extract_examples_per_testcase(
            xs, preds, confs, expect_results, labels, meta, nsamples,
            only_include_fail=True)
        if not result:
            return
        _, iters, _ = result
        for x, pred, conf, label, meta in iters:
            print(format_example_fn(x, pred, conf, label, meta))
        if type(preds) in [np.ndarray, list] and len(preds) > 1:
            print()
        print('----')

    def _fail_idxs(self, res_munch):
        return np.where(res_munch.passed == False)[0]

    def _filtered_idxs(self, res_munch):
        return np.where(res_munch.passed == None)[0]

    def _pass_idxs(self, res_munch):
        return np.where(res_munch.passed == True)[0]

    def _get_stats(self, core_record, res_munch):
        stats = Munch()
        n_run = n = len(core_record.data)
        if core_record.run_idxs is not None:
            n_run = len(core_record.run_idxs)
        fails = self._fail_idxs(res_munch).shape[0]
        filtered = self._filtered_idxs(res_munch).shape[0]
        nonfiltered = n_run - filtered
        stats.testcases = n
        if n_run != n:
            stats.testcases_run = n_run
        if filtered:
            stats.after_filtering = nonfiltered
            stats.after_filtering_rate = 100 * nonfiltered / n_run
        if nonfiltered != 0:
            stats.fails = fails
            stats.fail_rate = 100 * fails / nonfiltered
        return stats

    def _print_stats(self, core_record, res_munch):
        stats = self._get_stats(core_record, res_munch)
        print('Test cases:      %d' % stats.testcases)
        if 'testcases_run' in stats:
            print('Test cases run:  %d' % stats.testcases_run)
        if 'after_filtering' in stats:
            print('After filtering: %d (%.1f%%)' %
                  (stats.after_filtering, stats.after_filtering_rate))
        if 'fails' in stats:
            print('Fails (rate):    %d (%.1f%%)' %
                  (stats.fails, stats.fail_rate))

    def summary(
        self, core_record, res_munch, n=3, print_fn=None,
        format_example_fn=None, n_per_testcase=3, **kwargs
    ):
        """Print stats and example failures

        Parameters
        ----------
        n : int
            number of example failures to show
        print_fn : function
            If not None, use this to print a failed test case.
            Arguments: (xs, preds, confs, expect_results, labels=None,
                meta=None)
        format_example_fn : function
            If not None, use this to print a failed example within a test case
            Arguments: (x, pred, conf, label=None, meta=None)
        n_per_testcase : int
            Maximum number of examples to show for each test case
        """
        self._print_stats(core_record, res_munch)
        if not n:
            return
        if print_fn is None:
            print_fn = self._print

        def default_format_example(
                x, pred, conf, label, meta, *args, **kwargs):
            softmax = is_1d_list(conf)
            if softmax:
                if conf.shape[0] == 2:
                    conf = conf[1]
                    return '%.1f %s' % (conf, str(x))
                elif conf.shape[0] <= 4:
                    confs = ' '.join(['%.1f' % c for c in conf])
                    return '%s %s ' % (confs, str(x))

            if is_1d_list(pred):
                # sequence labeling
                return '%s %s' % (str(pred), str(x))
            else:
                conf = conf[pred]
                return '%s (%.1f) %s' % (pred, conf, str(x))

        if format_example_fn is None:
            format_example_fn = default_format_example

        fails = self._fail_idxs(res_munch)
        if fails.shape[0] == 0:
            return
        print()
        print('Example fails:')
        fails = np.random.choice(fails, min(fails.shape[0], n), replace=False)
        for f in fails:
            d_idx = f if core_record.run_idxs is None \
                    else core_record.run_idxs[f]
            # should be format_fn
            label, meta = core_record.label_meta(d_idx)
            # print(label, meta)

            print_fn(core_record.data[d_idx],
                     core_record.preds[d_idx],
                     core_record.confs[d_idx],
                     res_munch.results[f],
                     label, meta, format_example_fn, nsamples=n_per_testcase)

    def save_summary(self, core_record, **kwargs) -> None:
        pass


# ================== VISUAL SUMMARY

#     def _form_examples_per_testcase_for_viz(
#         self, xs, preds, confs, expect_results,
#         labels=None, meta=None, nsamples=3
#     ):
#         result = self._extract_examples_per_testcase(
#             xs, preds, confs, expect_results, labels, meta, nsamples,
#             only_include_fail=False)
#         if not result:
#             return []
#         idxs, iters, expect_results_sample = result
#         if not iters:
#             return []
#         start_idx = 1 if self.print_first else 0
#         if self.print_first:
#             base = iters[0]
#             try:
#                 conf = base[2][base[1]]
#             except Exception:
#                 conf = None
#             old_example = {
#                 "text": base[0], "pred": str(base[1]), "conf": conf}
#         else:
#             old_example = None

#         examples = []
#         for idx, e in enumerate(iters[start_idx:]):
#             try:
#                 conf = e[2][e[1]]
#             except Exception:
#                 conf = None
#             example = {
#                 "new": {"text": e[0], "pred": str(e[1]), "conf": conf},
#                 "old": old_example,
#                 "label": e[3],
#                 "succeed": int(expect_results_sample[start_idx:][idx] > 0)
#             }
#             examples.append(example)
#         return examples

#     def form_test_info(
#         self, name=None, description=None, capability=None
#     ):
#         n_run = len(self.data)
#         if self.run_idxs is not None:
#             n_run = len(self.run_idxs)
#         fails = self.fail_idxs().shape[0]
#         filtered = self.filtered_idxs().shape[0]
#         return {
#             "name": name if name else self.name,
#             "description": description if description else self.description,
#             "capability": capability if capability else self.capability,
#             "type": self.__class__.__name__.lower(),
#             "tags": [],
#             "stats": {
#                 "nfailed": fails,
#                 "npassed": n_run - filtered - fails,
#                 "nfiltered": filtered
#             }
#         }

#     def form_testcases(self, n_per_testcase=3):
#         testcases = []
#         nonfiltered_idxs = np.where(self.results.passed != None)[0]
#         for f in nonfiltered_idxs:
#             d_idx = f if self.run_idxs is None else self.run_idxs[f]
#             # should be format_fn
#             label, meta = self._label_meta(d_idx)
#             # print(label, meta)
#             succeed = self.results.passed[f]
#             if succeed is not None:
#                 examples = self._form_examples_per_testcase_for_viz(
#                     self.data[d_idx], self.results.preds[d_idx],
#                     self.results.confs[d_idx],
#                     self.results.expect_results[f],
#                     label, meta, nsamples=n_per_testcase)
#             else:
#                 examples = []
#             if examples:
#                 testcases.append({
#                     "examples": examples,
#                     "succeed": int(succeed),
#                     "tags": []
#                 })
#         return testcases

#     def visual_summary(
#         self, name=None, description=None, capability=None, n_per_testcase=3
#     ):
#         self._check_results()
#         # get the test meta
#         test_info = self.form_test_info(name, description, capability)
#         testcases = self.form_testcases(n_per_testcase)
#         return TestSummarizer(test_info, testcases)


###########################
#      SPECIFIC TESTS     #
###########################

class MFT(ExpectTest):
    def __init__(self, expect=None, agg_fn='all', labels=None, n_classes=None):
        self.labels = labels
        self.n_classes = n_classes
        super().__init__(expect, agg_fn=agg_fn, print_first=False, name="MFT")
        self._group_flatten_method = FlattenGroup.FLATTEN_ALL

    @overrides
    def compute(self, core_record) -> Munch:
        if self.labels is not None:
            core_record.labels = self.labels

        elif core_record.labels is not None and self.n_classes is not None:
            # do this temporarily -- to get MFT results
            core_record.adjust_for_class_eval(
                self.n_classes, filter_none=False)

        if core_record.labels is None and self.expect is None:
            raise(Exception('Must specify either \'expect\' or \'labels\''))
        if core_record.labels is not None and self.expect is None:
            self.expect = Expect.eq()

        to_ret = super().compute(core_record)
        return to_ret


class INV(ExpectTest):
    def __init__(self, expect=None, threshold=0, agg_fn='all_except_first'):
        if expect is None:
            expect = Expect.inv(threshold, allow_list_examples=True)
        super().__init__(expect, agg_fn=agg_fn, print_first=True, name="INV")


class DIR(ExpectTest):
    def __init__(self, expect, agg_fn='all_except_first'):
        super().__init__(expect, agg_fn=agg_fn, print_first=True, name="DIR")
