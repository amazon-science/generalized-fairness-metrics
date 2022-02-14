import sys
sys.path.insert(0, "../")
sys.path.insert(0, ".")

from expanded_checklist.checklist.editor import Editor
from src.tests.fill_the_lexicon import fill_the_lexicon
from src.tests.test_model import quick_test, test_model_on_saved_suite
from src.tests.save_and_load import *
import src.tests.test_creation_utils as test_utils

from expanded_checklist.checklist.tests import *


# NOTE: FOR EXAMPLES OF PLOTTING SEE Examples.ipynb in notebooks
# That notebook also contains all the examples in this script.


# 1. CREATING A TEST SUITE
# commented out because we use an existing suite for which we already have
# model's predictions. See src/tests/test_suites for the functions used to
# create the suites. Especially explicit_terms_suites.py:

# editor = Editor()
# fill_the_lexicon(editor)
# suite, suite_name, data = get_names_suite(editor, nsamples=100)

# we don't create as suite, but use an existing one:
suite_name = "names_gender"


# 2. DEFINING METRICS
# we create a range of task-agnostic tests
# see checklist/tests/__init__.py
all_tests = get_all_tests()


# 3. TESTING A BINARY CLASSIFIER
# it returns a TestSuite object with the given name:
# this object bundles a number of EvalCores
# (at the moment all bias suites have one core per test suite,
# and it has the same name as the suite).
suite = test_model_on_saved_suite(
    suite_name=suite_name,
    mname='roberta-semeval-2',
    tests=all_tests,
    task="SENT"
)

# We could also filter the evaluation data based on e.g. domain, if such
# field is defined in the templates; we can filter based on any field/column
# name in the templates csv, e.g. we could filter based on SENT
# (sentiment classification class)
# NOTE: the set of templates for names_gender suite doesn't include DOMAIN
# key so this code is commented out.
# test_model_on_saved_suite(
#     suite_name=suite_name,
#     mname='roberta-sst-2',
#     tests=all_tests,
#     task="SENT",
#     data_filter={"DOMAIN": "movie"}
# )


# 4. TESTING A 3-CLASS CLASSIFIER
# Some tests won't work, like aeg because they use the output of sigmoid in the
# calculations; aeg's extended version will work because it uses the prob. of
# specific class for which the example belongs to.
suite2 = test_model_on_saved_suite(
    suite_name=suite_name,
    mname='roberta-semeval-3',
    tests=all_tests,
    task="SENT"
)


# 4. TESTING NER
# Some metrics based on FPR and TNR won't work (e.g FPED).
# Basic classification metric also won't work.
suite3 = test_model_on_saved_suite(
    suite_name=suite_name,
    mname='ner-roberta-conll2003',
    tests=all_tests,
    task="NER"
)
