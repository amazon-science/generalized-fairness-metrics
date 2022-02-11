import logging
import os
import re

src_dir = f"{os.path.dirname(os.path.realpath(__file__))}"
ROOT = re.sub(r"/src$", "", src_dir)

dev_semeval_path =\
    f"{ROOT}/datasets/semeval-v-oc/2018-Valence-oc-En-dev.txt"
test_semeval_path =\
    f"{ROOT}/datasets/semeval-v-oc/2018-Valence-oc-En-test-gold.txt"
train_semeval_path =\
    f"{ROOT}/datasets/semeval-v-oc/2018-Valence-oc-En-train.txt"

dev_conll_path = f"{ROOT}/datasets/conll2003/testa.txt"
test_conll_path = f"{ROOT}/datasets/conll2003/testb.txt"
train_conll_path = f"{ROOT}/datasets/conll2003/train.txt"

identity_terms_dir_path = f"{ROOT}/terms/identity_terms"

test_suites_path = f"{ROOT}/test_suites"

predictions_path = f"{ROOT}/predictions"
samples_path = f"{ROOT}/samples"

templates_path = f"{ROOT}/templates"
nationality_template_path = f"{templates_path}/nationality_templates.csv"
generic_template_path = f"{templates_path}/generic_templates.csv"
generic_template_path_adj_or_pp =\
    f"{templates_path}/generic_templates_adj_or_pp.csv"
gend_sex_template_path = f"{templates_path}/gender_sexuality_templates.csv"
religion_template_path = f"{templates_path}/religion_templates.csv"
people_template_path = f"{templates_path}/people_templates.csv"
disability_template_path = f"{templates_path}/disability_templates.csv"
age_template_path = f"{templates_path}/age_templates.csv"
ethnicity_template_path = f"{templates_path}/ethnicity_templates.csv"

log_level = logging.INFO

# for those tasks we have columns in our templates
supported_tasks = ["NER", "SENT"]

seq_tasks = ["NER"]

# transformers crash on sequences that are too long
max_position_embeddings = 512

if not os.path.isdir(test_suites_path):
    os.makedirs(test_suites_path)

if not os.path.isdir(predictions_path):
    os.makedirs(predictions_path)

if not os.path.isdir(samples_path):
    os.makedirs(samples_path)
