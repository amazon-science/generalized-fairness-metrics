
import collections
from collections import defaultdict, OrderedDict
import dill
import json

from .utils import load_test, read_pred_file, sequence_tasks, tokenize, \
    TOKENIZATION_DICT

from .viewer.suite_summarizer import SuiteSummarizer

import logging
logger = logging.getLogger(__name__)


class TestSuite:
    def __init__(self, format_example_fn=None, print_fn=None):
        # TODO: rename the test suite fields to evaluation cores.
        self.tests = OrderedDict()
        self.info = defaultdict(lambda: defaultdict(lambda: ''))
        self.format_example_fn = format_example_fn
        self.print_fn = print_fn
        self.test_ranges = {}

    @staticmethod
    def from_file(path):
        """Loads suite from file

        Parameters
        ----------
        path : string
            pickled (dill) file

        Returns
        -------
        TestSuite
            the suite

        """
        return load_test(path)

    def set_tests(self, tests):
        for name, test in self.tests.items():
            test.set_tests(tests)

    def add(
        self, test, name=None, capability=None, description=None,
        format_example_fn=None, print_fn=None, overwrite=False
    ):
        """Adds a test to suite

        Parameters
        ----------
        test :
            test
        name : string
            test name. If test has test.name, this is optional.
        capability : string
            test capability. If test has test.capability, this is optional.
        description : string
            test description. If test has test.capability, this is optional.
        format_example_fn : function
            If not None, use this to print a failed example within a test case
            Arguments: (x, pred, conf, label=None, meta=None)
        print_fn : function
            If not None, use this to print a failed test case.
            Arguments: (xs, preds, confs, expect_results, labels=None, meta=None)
        overwrite : bool
            If False, will raise exception if test with same name is already in suite.

        """
        if name is None and test.name is None:
            raise(Exception(
                'If test does not have test.name, you must specify a name'))
        if capability is None and test.capability is None:
            raise(Exception(
                'If test does not have test.capabiliy, you must specify a capability'))
        if name is None:
            name = test.name
        if capability is None:
            capability = test.capability
        if description is None:
            description = test.description
        if name in self.tests and not overwrite:
            raise(Exception('There is already a test named %s suite. Run with overwrite=True to overwrite' % name))
        if name in self.info:
            del self.info[name]

        typez = test.name
        self.tests[name] = test
        self.info[name]['capability'] = capability
        self.info[name]['type'] = typez
        if description:
            self.info[name]['description'] = description
        if format_example_fn:
            self.info[name]['format_example_fn'] = format_example_fn
        if print_fn:
            self.info[name]['print_fn'] = format_example_fn

    def remove(self, name):
        """Removes test from suite

        Parameters
        ----------
        name : string
            test name

        """
        if name not in self.tests:
            raise(Exception('%s not in suite.' % name))
        del self.tests[name]
        del self.info[name]

    def to_dict(self, example_to_dict_fn=None, n=None, seed=None, new_sample=False):
        if example_to_dict_fn is None:
            try:
                example_to_dict_fn = self.example_to_dict_fn
            except AttributeError:
                raise(Exception('suite does not have example_to_dict_fn, must pass function as argument.'))
        examples = self.get_raw_examples(format_fn=lambda x:x, n=n, seed=seed, new_sample=new_sample)
        data_keys = list(example_to_dict_fn(examples[0]).keys())
        keys = data_keys + ['test_name', 'test_case', 'example_idx']
        hf_dict = { k:[] for k in keys }
        for e in examples:
            m = example_to_dict_fn(e)
            for k,v  in m.items():
                hf_dict[k].append(v)
        for test_name, r in sorted(self.test_ranges.items(), key=lambda x:x[1][0]):
            test = self.tests[test_name]
            size = r[1] - r[0]
            hf_dict['test_name'].extend([test_name for _ in range(size)])
            hf_dict['test_case'].extend(test.result_indexes)
            cnt = collections.defaultdict(lambda: 0)
            example_idx = []
            for i in test.result_indexes:
                example_idx.append(cnt[i])
                cnt[i] += 1
            hf_dict['example_idx'].extend(example_idx)
        return hf_dict

    def get_raw_examples(self, file_format=None, format_fn=None, n=None, seed=None, new_sample=True):
        if new_sample or len(self.test_ranges) == 0:
            self.test_ranges = {}
            all_examples = self.create_raw_example_list(file_format=file_format, format_fn=format_fn, n=n, seed=seed)
        else:
            all_examples = self.get_raw_example_list(file_format=file_format, format_fn=format_fn)
        return all_examples

    def get_raw_example_list(self, file_format=None, format_fn=None):
        if not self.test_ranges:
            raise(Exception('example list not created. please call create_raw_example_list, or to_raw_file first'))
        examples = []
        for test_name, r in sorted(self.test_ranges.items(), key=lambda x:x[1][0]):
            test = self.tests[test_name]
            test_examples = test.to_raw_examples(file_format=file_format, format_fn=format_fn,
                                         n=None, seed=None, new_sample=False)
            assert len(test_examples) == r[1] - r[0]
            examples.extend(test_examples)
        return examples

    def create_raw_example_list(self, file_format, format_fn, n, seed):
        self.test_ranges = {}
        current_idx = 0
        all_examples = []
        for name, t in self.tests.items():
            examples = t.to_raw_examples(file_format=file_format, format_fn=format_fn, n=n, seed=seed, new_sample=True)
            self.test_ranges[name] = (current_idx, current_idx + len(examples))
            current_idx += len(examples)
            all_examples.extend(examples)
        return all_examples


    def to_raw_file(self, path, task, file_format=None, format_fn=None, header=None, n=None, seed=None, new_sample=True):
        """Flatten all tests into individual examples and print them to file.
        Indices of example to test case will be stored in each test.
        If n is not None, test.run_idxs will store the test case indexes.
        The line ranges for each test will be saved in self.test_ranges.

        Parameters
        ----------
        path : string
            File path
        file_format : string, must be one of 'jsonl', 'squad', 'qqp_test', or None
            None just calls str(x) for each example in self.data
            squad assumes x has x['question'] and x['passage'], or that format_fn does this
        format_fn : function or None
            If not None, call this function to format each example in self.data
        header : string
            If not None, first line of file
        n : int
            If not None, number of samples to draw
        seed : int
            Seed to use if n is not None
        new_sample: bool
            If False, will rely on a previous sample and ignore the 'n' and 'seed' parameters

        """
        ret = ''
        all_examples = []
        add_id = False

        if file_format == 'qqp_test':
            add_id = True
            file_format = 'tsv'
            header = 'id\tquestion1\tquestion2'
        if header is not None:
            ret += header.strip('\n') + '\n'

        if task in sequence_tasks:
            if format_fn is not None:
                logger.warning("Replacing given format_fn with a tokenizer.")

            # if the data is pre-tokenized and this is recorded in the
            # metadata, use that tokenization (this ensures the per-token
            # labels match the predictions from the model)
            format_fn =\
                lambda x, m: "\n".join(m[TOKENIZATION_DICT][x]) + "\n" \
                    if TOKENIZATION_DICT in m \
                    else "\n".join(tokenize(x)) + "\n"

        all_examples = self.get_raw_examples(file_format=file_format, format_fn=format_fn, n=n, seed=seed, new_sample=new_sample)

        if add_id and file_format == 'tsv':
            all_examples = ['%d\t%s' % (i, x) for i, x in enumerate(all_examples)]
        if file_format == 'squad':
            ret_map = {'version': 'fake',
                       'data': []}
            for i, x in enumerate(all_examples):
                r = {'title': '',
                     'paragraphs': [{
                        'context': x['passage'],
                        'qas': [{'question' : x['question'],
                                 'id': str(i)
                                 }]
                      }]
                    }
                ret_map['data'].append(r)
            ret = json.dumps(ret_map)
        else:
            ret += '\n'.join(all_examples)
        with open(path, 'w') as f:
            f.write(ret)

    def run_from_preds_confs(
            self, preds, confs, label_vocab=None, task=None, data_filter=None, overwrite=False):
        for n, t in self.tests.items():
            p = preds[slice(*self.test_ranges[n])]
            c = confs[slice(*self.test_ranges[n])]
            t.run_from_preds_confs(
                p, c, label_vocab=label_vocab, task=task,
                data_filter=data_filter, overwrite=overwrite)

    def run_from_file(self, path, file_format=None, format_fn=None, task=None,
            data_filter=None, ignore_header=False, overwrite=False):
        """Update test.results (run tests) for every test, from a prediction file

        Parameters
        ----------
        path : string
            prediction file path
        file_format : string
            None, or one of 'pred_only', 'softmax', binary_conf', 'pred_and_conf', 'pred_and_softmax', 'squad',
            pred_only: each line has a prediction
            softmax: each line has prediction probabilities separated by spaces
            binary_conf: each line has the prediction probability of class 1 (binary)
            pred_and_conf: each line has a prediction and a confidence value, separated by a space
            pred_and_softmax: each line has a prediction and all softmax probabilities, separated by a space
            squad: TODO
        format_fn : function
            If not None, function that reads a line in the input file and outputs a tuple of (prediction, confidence)
        ignore_header : bool
            If True, skip first line in the file
        overwrite : bool
            If False, raise exception if results already exist

        """
        pred_file_out = read_pred_file(path, file_format=file_format,
                                       format_fn=format_fn,
                                       ignore_header=ignore_header)

        if len(pred_file_out) == 2:
            preds, confs = pred_file_out
            label_vocab = None
        elif len(pred_file_out) == 3:
            preds, confs, label_vocab = pred_file_out
        else:
            raise Exception('Incorrect output of the read_pred_file function.')

        self.run_from_preds_confs(
            preds, confs, label_vocab, task=task, data_filter=data_filter, overwrite=overwrite)

    # TODO: add support for this
    # def run(self, predict_and_confidence_fn, verbose=True, **kwargs):
    #     """Runs all tests in the suite
    #
    #     Parameters
    #     ----------
    #     predict_and_confidence_fn : function
    #         Takes as input a list of examples
    #         Outputs a tuple (predictions, confidences)
    #     overwrite : bool
    #         If False, raise exception if results already exist
    #     verbose : bool
    #         If True, print extra information
    #     n : int
    #         If not None, number of samples to draw
    #     seed : int
    #         Seed to use if n is not None

    #     """
    #     for n, t in self.tests.items():
    #         if verbose:
    #             print('Running', n)
    #         t.run(predict_and_confidence_fn, verbose=verbose, **kwargs)

    def summary(self, types=None, capabilities=None, **kwargs):
        """Print stats and example failures for each test.

        Parameters
        ----------
        types : list(string)
            If not None, will only show tests of these test types.
            Options are MFT, INV, and DIR
        capabilities : list(string)
            If not None, will only show tests with these capabilities.
        **kwargs : type
            Will be passed as arguments to each test.summary()

        """
        vals = collections.defaultdict(lambda: 100, {'MFT': 0, 'INV': 1, 'DIR': 2})
        tests = self.tests.keys()
        capability_order = ['Vocabulary', 'Taxonomy', 'Robustness', 'NER',  'Fairness', 'Temporal', 'Negation', 'Coref', 'SRL', 'Logic']
        cap_order = lambda x:capability_order.index(x) if x in capability_order else 100
        caps = sorted(set([x['capability'] for x in self.info.values()]), key=cap_order)
        for capability in caps:
            if capabilities is not None and capability not in capabilities:
                continue
            print(capability)
            print()
            tests = [x for x in self.tests if self.info[x]['capability'] == capability]
            for n in tests:
                if types is not None and self.info[n]['type'] not in types:
                    continue

                nspaces = 5
                print_n = "Eval Core: " + n
                bar = "#" * (len(print_n) + nspaces * 2 + 2)
                print(bar)
                print("#" + " " * (len(bar)-2) + "#")
                print("#" + " " * nspaces + print_n + " " * nspaces + "#")
                print("#" + " " * (len(bar)-2) + "#")
                print(bar)

                if 'format_example_fn' not in kwargs:
                    kwargs['format_example_fn'] = self.info[n].get('format_example_fn', self.format_example_fn)
                if 'print_fn' not in kwargs:
                    kwargs['print_fn'] = self.info[n].get('print_fn', self.print_fn)
                self.tests[n].summary(**kwargs)
                print()
                print()
            print()
            print()

    def get_result_dict(self):
        name2result_dict = {}
        for n, t in self.tests.items():
            name2result_dict[n] = t.results
        return name2result_dict


    # TODO: visual summary is unsupported atm
    # def visual_summary_by_test(self, testname):
    #     """Displays visual summary for a single test.

    #     Parameters
    #     ----------
    #     testname : string
    #         name of the test

    #     Returns
    #     -------
    #     test.visual_summary
    #         summary

    #     """
    #     if not testname in self.tests:
    #         raise(Exception(f"There's no test named {testname} in the suite!"))
    #     test, info = self.tests[testname], self.info[testname]
    #     return test.visual_summary(
    #         name=testname,
    #         capability=info["capability"] if "capability" in info else None,
    #         description=info["description"] if "description" in info else None
    #     )

    # def _on_select_test(self, testname: str):
    #     if not testname:
    #         test_info, testcases = {}, []
    #     else:
    #         if not testname in self.tests:
    #             raise(Exception(f"There's no test named {testname} in the suite!"))
    #         test, info = self.tests[testname], self.info[testname]
    #         test_info = test.form_test_info(
    #             name=testname,
    #             capability=info["capability"] if "capability" in info else None,
    #             description=info["description"] if "description" in info else None
    #         )
    #         n = 1 if self.info[testname]['type'] == 'MFT' else 2
    #         testcases = test.form_testcases(n_per_testcase=n)
    #     return test_info, testcases

    # def visual_summary_table(self, types=None, capabilities=None):
    #     """Displays a matrix visualization of the test suite

    #     Parameters
    #     ----------
    #     types : list(string)
    #         If not None, will only show tests of these test types.
    #         Options are MFT, INV, and DIR
    #     capabilities : list(string)
    #         If not None, will only show tests with these capabilities.

    #     Returns
    #     -------
    #     SuiteSummarizer
    #         jupyter visualization

    #     """
    #     logger.info("Please wait as we prepare the table data...")
    #     test_infos = []
    #     for testname in self.tests.keys():
    #         test, info = self.tests[testname], self.info[testname]

    #         local_info = test.form_test_info(
    #             name=testname,
    #             capability=info["capability"] if "capability" in info else None,
    #             description=info["description"] if "description" in info else None
    #         )
    #         if (not capabilities or local_info["capability"] in capabilities) and \
    #             (not types or local_info["type"] in types):
    #             test_infos.append(local_info)

    #     capability_order = ['Vocabulary', 'Taxonomy', 'Robustness', 'NER',  'Fairness', 'Temporal', 'Negation', 'Coref', 'SRL', 'Logic']
    #     cap_order = lambda x: capability_order.index(x["capability"]) if x in capability_order else 100
    #     test_infos = sorted(test_infos, key=cap_order)
    #     return SuiteSummarizer(
    #         test_infos=test_infos,
    #         select_test_fn=self._on_select_test
    #     )

    def save(self, path):
        """Serializes the suite and saves it to a file

        Parameters
        ----------
        path : string
            output file path

        """
        with open(path, 'wb') as f:
            dill.dump(self, f, recurse=True)
