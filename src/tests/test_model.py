import src.config as cfg
import subprocess
import os
import time
from typing import Union, Tuple, Any, Callable, Dict

from expanded_checklist.checklist.test_suite import TestSuite
from expanded_checklist.checklist.eval_core import EvaluationCore

from src.tests.save_and_load import load_suite

import filecmp
import logging

logging.basicConfig(level=cfg.log_level)
logger = logging.getLogger(__name__)

# seed is kept constant to always have the same samples for given n
SEED = 1


def _get_dataset_name(model_name: str) -> str:
    ds_names = ['sst-2', 'sst-3', 'semeval-2', 'semeval-3', 'conll2003']
    for dsn in ds_names:
        if f"-{dsn}" in model_name:
            return dsn
    return None


def _get_preds_path(
    mname: str,
    test_name: str,
    num_samples: str  # either a str(int) or 'full',
) -> Tuple[str, str]:
    out_sub_dir = _get_dataset_name(mname)

    # sample sentences are the same, no matter what model is tested
    return f"{cfg.predictions_path}/{out_sub_dir}/preds-{mname}-" +\
        f"{test_name}-{num_samples}.txt"


def _get_samples_path(
    test_name: str,
    task: str,
    num_samples: str  # either a str(int) or 'full'
):
    return \
        f"{cfg.samples_path}/samples-{test_name}-{task}-{num_samples}.txt"


def get_predictions(
    mname: str,
    inputs_path: str,
    preds_path: str,
) -> Any:
    logger.info(f"Getting {mname} predictions (may take a while) ...")
    bashCommand =\
        f"bash {cfg.ROOT}/get_predictions.sh exp={mname} " +\
        f"data={inputs_path} out_file={preds_path}"

    logger.info("\nExecuting bash command:")
    logger.info(bashCommand)

    process = subprocess.Popen(
        bashCommand.split(), stdout=subprocess.PIPE, cwd=cfg.ROOT)
    output, error = process.communicate()
    return output, error


def test_model_on_saved_suite(
    suite_name: str,
    mname: str,
    # task determines what labels are used and what preds are expected
    task: str,
    nsamples: int = None,
    repred: bool = False,
    tests: list = None,
    get_predictions_fun: Callable = None,
    data_filter: Dict = None  # e.g. {"DOMAIN": "business"}
) -> None:
    """
    This is the function to use in order to evaluate different models on
    the sample sampled data. It operates on a name of the saved suite -- which
    ensures that exactly the same suite is used every time and it uses a set
    SEED which ensures deterministic samples.

    NOTE: for now this funcion is not robust to the suite being overwritten
    in a pickle file. It will throw an assertion error.
    """
    if task not in cfg.supported_tasks:
        raise Exception(f'Unsupported task: {task}.')

    suite: TestSuite = load_suite(suite_name)
    if tests:
        suite.set_tests(tests)

    str_nsamples = str(nsamples) if nsamples else "full"
    preds_path = _get_preds_path(mname, suite_name, str_nsamples)
    samples_path = _get_samples_path(suite_name, task, str_nsamples)
    sent_samples_path = _get_samples_path(suite_name, "SENT", str_nsamples)

    # the suite has already sampled sentences before for that nsamples,
    # we call to_raw_file to (i) set all the necessary variables in the test
    # suite, e.g. result_indexes and to (ii) check if those indexes correspond
    # to the examples saved in samples_path
    if os.path.isfile(samples_path):
        logging.info(f"Sample already exists, checking that suite matches " +
                     "the sample...")
        tmp_inputs = "/tmp/tmp_inputs.txt"
        # compare based on SENT task examples (it's more efficient, esp.
        # tokenized inputs are time consuming to get)
        suite.to_raw_file(tmp_inputs, task="SENT", n=nsamples, seed=SEED)
        assert filecmp.cmp(tmp_inputs, sent_samples_path)
        os.remove(tmp_inputs)
    else:
        logging.info(f"Sample doesn't exist for the suite for that task, " +
                     "generating sample...")
        # different models use different samples because they may expect data
        # in different formats (e.g. NER expects pre-tokenised data)
        # note that as a result, when using sampling, data from the same suite
        # can be slightly different for different tasks
        suite.to_raw_file(samples_path, task=task, n=nsamples, seed=SEED)
        if task != "SENT":
            suite.to_raw_file(
                sent_samples_path, task="SENT", n=nsamples, seed=SEED)
        repred = True

    if not repred and os.path.isfile(preds_path):
        logger.info(f"{mname} predictions already exist.")
    else:
        if get_predictions_fun is None:
            output, error = get_predictions(mname, samples_path, preds_path)
            if not os.path.isfile(preds_path):
                logging.error(f"Could not get predictions for {mname}.")
                logger.error("ERROR:\n" + str(error))
                logger.error("OUTPUT:\n" + str(output))
        else:
            get_predictions_fun(mname, samples_path, preds_path)

    logger.info(f">>> Testing {mname} ...")
    if task in ["NER"]:
        file_format = 'seq_pred_and_softmax'
    else:
        file_format = 'pred_and_softmax'

    suite.run_from_file(
        preds_path, file_format=file_format,
        task=task, data_filter=data_filter, overwrite=True)
    suite.summary()
    return suite


def quick_test(
    mname: str,
    test: Union[EvaluationCore, TestSuite],
    task: str,
    nsamples: int = None,
    repred: bool = False,
    get_predictions_fun=None,
    tests: list = None,
    data_filter: Dict = None  # e.g. {"DOMAIN": "business"}
) -> None:
    """
    This function should be use for quick tests of single models.
    For comparison of different models on the same suite, it's safer to use
    test_model_on_saved_suite function which performs additional checks to
    ensure all models are tested on exactly the same samples of tests.
    """
    if task not in ["NER", "SENT"]:
        raise Exception(f'Unsupported task: {task}.')

    if tests:
        test.set_tests(tests)

    ts = time.time()
    samples_path = f"/tmp/tmp_inputs_{ts}.txt"
    preds_path = f"/tmp/tmp_preds_{ts}.txt"
    test.to_raw_file(samples_path, task=task, n=nsamples, seed=SEED)

    if get_predictions_fun is None:
        output, error = get_predictions(mname, samples_path, preds_path)
        if not os.path.isfile(preds_path):
            logging.error(f"Could not get predictions for {mname}.")
            logger.error(error)
            logger.error(output)
    else:
        get_predictions_fun(mname, samples_path, preds_path)

    if task in cfg.seq_tasks:
        file_format = 'seq_pred_and_softmax'
    else:
        file_format = 'pred_and_softmax'

    logger.info(f">>> Testing {mname} ...")
    if task in ["NER"]:
        file_format = 'seq_pred_and_softmax'
    else:
        file_format = 'pred_and_softmax'
    test.run_from_file(
        preds_path, file_format=file_format,
        task=task, data_filter=data_filter, overwrite=True)

    os.remove(preds_path)
    os.remove(samples_path)
    test.summary()
