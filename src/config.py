import logging

ROOT = "/home/ubuntu/workplace/ComprehendBiasTools"
dev_sst_path = f"{ROOT}/datasets/SST/trees/dev.txt"
test_sst_path = f"{ROOT}/datasets/SST/trees/test.txt"
train_sst_path = f"{ROOT}/datasets/SST/trees/train.txt"

dev_semeval_path =\
    f"{ROOT}/datasets/SemEval2018-Task1-all-data/English/V-oc/2018-Valence-oc-En-dev.txt"
test_semeval_path =\
    f"{ROOT}/datasets/SemEval2018-Task1-all-data/English/V-oc/2018-Valence-oc-En-test-gold.txt"
train_semeval_path =\
    f"{ROOT}/datasets/SemEval2018-Task1-all-data/English/V-oc/2018-Valence-oc-En-train.txt"

dev_conll_path = f"{ROOT}/datasets/conll2003/valid.txt"
test_conll_path = f"{ROOT}/datasets/conll2003/test.txt"
train_conll_path = f"{ROOT}/datasets/conll2003/train.txt"

identity_terms_dir_path = f"{ROOT}/terms/identity_terms"
helper_terms_dir_path = f"{ROOT}/terms/helper_terms"
test_suites_path = f"{ROOT}/test_suites"

gendered_words_path = f"{ROOT}/gendered_words/gendered_words.json"


main_template_path = f"{ROOT}/templates/main_templates.csv"


nationality_template_path = f"{ROOT}/templates/nationality_templates.csv"
generic_template_path = f"{ROOT}/templates/generic_templates.csv"
generic_template_path_adj_or_pp =\
    f"{ROOT}/templates/generic_templates_adj_or_pp.csv"
gend_sex_template_path = f"{ROOT}/templates/gender_sexuality_templates.csv"
religion_template_path = f"{ROOT}/templates/religion_templates.csv"
people_template_path = f"{ROOT}/templates/people_templates.csv"
disability_template_path = f"{ROOT}/templates/disability_templates.csv"
age_template_path = f"{ROOT}/templates/age_templates.csv"
ethnicity_template_path = f"{ROOT}/templates/ethnicity_templates.csv"

emotion_terms_path =\
    f"{helper_terms_dir_path}/emotions_and_polarity_terms.csv"


log_level = logging.INFO

domains = ['all']  #['all', 'movie', 'personal', 'culture', 'education', 'professional']

# for those tasks we have columns in our templates
supported_tasks = ["NER", "SENT"]

seq_tasks = ["NER"]

# transformers crash on sequences that are too long
max_position_embeddings = 512