
import os
import regex as re
import dill
import glob
import os

from expanded_checklist.checklist.test_suite import TestSuite
from expanded_checklist.checklist.tests import get_all_tests
from expanded_checklist.checklist.graphs.graphs import plot_all_tests

import src.config as cfg
import logging

logging.basicConfig(level=cfg.log_level)
logger = logging.getLogger(__name__)


##########################################
#   SAVING/LOADING SUITES/SAVED RESULTS  #
##########################################

def save_suite(
    suite: TestSuite,
    name: str
) -> None:

    if type(suite) == TestSuite:
        for test in suite.tests:
            suite.tests[test].name = test
            suite.tests[test].description = suite.info[test]['description']
            suite.tests[test].capability = suite.info[test]['capability']

    path = f'{cfg.test_suites_path}/{name}.pkl'
    logger.info(f"Saving test suite in {path}")

    # clean all samples for a suite with that name
    to_rem = glob.glob(f"{cfg.samples_path}/samples-{name}-*.txt")

    for i in range(len(to_rem)):
        x = to_rem[i]
        samp = re.search(r'samples-(.*)\.txt', x)
        if samp:
            samp = samp.group(1)
            for task in cfg.supported_tasks:
                samp = samp.replace(f"-{task}", "")
            to_rem += glob.glob(f"{cfg.predictions_path}/*/preds-*-{samp}.txt")

    logger.warning(
        f"Deleting existing samples/predictions for suite {name}!")
    for fpath in to_rem:
        try:
            os.remove(fpath)
        except Exception:
            logger.error(f"Error while deleting file: {fpath}")
    suite.save(path)


def load_suite(
    name: str
) -> TestSuite:
    path = f'{cfg.test_suites_path}/{name}.pkl'
    return TestSuite.from_file(path)


def save_results(name2result_dict, model_name):
    if not os.path.exists(f'{cfg.ROOT}/saved_results'):
        os.makedirs(f'{cfg.ROOT}/saved_results')

    for name, result_dict in name2result_dict.items():
        with open(f'{cfg.ROOT}/saved_results/{name}-{model_name}.pkl',
                  'wb') as f:
            dill.dump(result_dict, f, recurse=True)


def load_results(core_name, model_name):
    dill._dill._reverse_typemap['ClassType'] = type

    fname = f'{cfg.ROOT}/saved_results/{core_name}-{model_name}.pkl'
    if not os.path.exists(fname):
        return None

    with open(fname, 'rb') as f:
        to_ret = dill.load(f)
    return to_ret


def load_and_plot(
        suite_name, model_name, cl, scaling=1,
        counterfactual=False, plots_dir=f'{cfg.ROOT}/plots',
        skip_group_pcms=True):
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    result_dict = load_results(suite_name, model_name)

    return plot_all_tests(
        model_name, result_dict, suite_name,
        out_dir=plots_dir, cl=cl, counterfactual=counterfactual,
        scaling=scaling, skip_group_pcms=skip_group_pcms)
