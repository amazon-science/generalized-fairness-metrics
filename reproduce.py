from expanded_checklist.checklist.editor import Editor
from src.tests.fill_the_lexicon import fill_the_lexicon
from src.tests.test_model import test_model_on_saved_suite

from src.tests.test_suites.explicit_terms_suites import *
import src.config as cfg
from expanded_checklist.checklist.tests import *
from expanded_checklist.checklist.graphs.graphs import *
from src.tests.save_and_load import *
import pandas as pd
import argparse
import os


attribute_names = {
    'grouped_religion': 'religion',
    'grouped_sexuality': 'sexuality',
    'grouped_gender': 'gender',
    'grouped_disability': 'disability',
    'grouped_race': 'race',
    'grouped_age': 'age',
    'grouped_country_by_gdp_ppp_quantile': 'nationality',
    'names_gender': 'names'
}


def run_and_save_res(suite_name, model_name, task="SENT", tests=None):
    if tests is None:
        tests = get_all_tests()

    suite = test_model_on_saved_suite(
        suite_name=suite_name,
        mname=model_name,
        tests=tests,
        task=task
    )

    name2result_dict = suite.get_result_dict()
    save_results(name2result_dict, model_name)


def create_all_suites(editor):
    get_religion_suite(editor, nsamples=None)
    get_sexuality_suite(editor, nsamples=None)
    get_gender_suite(editor, nsamples=None)
    get_names_suite(editor, nsamples=100)
    get_nationality_suite(editor, group_key="GDP_PPP", nsamples=None)
    get_disability_suite(editor)
    get_race_suite(editor)
    get_age_suite(editor)


def get_results_for_model(mname, suite_names, task="SENT"):
    for sn in suite_names:
        run_and_save_res(sn, mname, task=task)


def _plot_multiple_attributes(mname, cl, plots_dir):
    core_name2test2res = {}
    for name, clean_name in attribute_names.items():
        if 'grouped' in name:
            core_name2test2res[clean_name] = load_results(name, mname)

    return plot_all_tests_for_different_cores(
        model_name=mname,
        core_name2test2res=core_name2test2res,
        out_dir=plots_dir,
        cl=cl
    )


def plot_scaled_with_another_result(
        df_to_plot, extra_df, metric_names,
        names_to_rem, plot_name, plots_dir, tight=True):
    df_for_scaling = extra_df.copy()
    df_for_scaling.rename(columns=lambda x: x + "_DROP", inplace=True)

    concat_df = pd.concat([df_to_plot, df_for_scaling], axis=1)

    metric_names = metric_names + [x + "_DROP" for x in names_to_rem]
    plot_groups(
        concat_df, metric_names, plot_name,
        out_dir=plots_dir, scaling=Scaling.MAX_ABS,
        tight=tight)


def plot_multiple_attributes(plots_dir):
    se2_df, se2_mnames = _plot_multiple_attributes(
        "roberta-semeval-2", 1, plots_dir=plots_dir)
    se3_df, se3_mnames = _plot_multiple_attributes(
        "roberta-semeval-3", 2, plots_dir=plots_dir)

    plot_scaled_with_another_result(
        se2_df, se3_df, se2_mnames, se3_mnames,
        "all-attrs-semeval-2-scaled-with-semeval-3.pdf", plots_dir=plots_dir)
    plot_scaled_with_another_result(
        se3_df, se2_df, se3_mnames, se2_mnames,
        "all-attrs-semeval-3-scaled-with-semeval-2.pdf", plots_dir=plots_dir)


def plot_gender_and_names(plots_dir):
    # plotting gender and names
    names_df_se2, names_names_se2 =\
        load_and_plot(
            "names_gender", "roberta-semeval-2", cl=1, plots_dir=plots_dir)
    gender_df_se2, gender_names_se2 =\
        load_and_plot(
            "grouped_gender", "roberta-semeval-2", cl=1, plots_dir=plots_dir)

    plot_scaled_with_another_result(
        gender_df_se2, names_df_se2, gender_names_se2, names_names_se2,
        "semeval_2_grouped_gender_normalized_with_names.pdf",
        tight=False, plots_dir=plots_dir)

    plot_scaled_with_another_result(
        names_df_se2, gender_df_se2, names_names_se2, gender_names_se2,
        "semeval_2_names_gender_normalized_with_grouped_gender.pdf",
        tight=False, plots_dir=plots_dir)

    names_df_se3, names_names_se3 =\
        load_and_plot(
            "names_gender", "roberta-semeval-3", cl=2, plots_dir=plots_dir)
    gender_df_se3, gender_names_se3 =\
        load_and_plot(
            "grouped_gender", "roberta-semeval-3", cl=2, plots_dir=plots_dir)

    plot_scaled_with_another_result(
        gender_df_se3, names_df_se3, gender_names_se3, names_names_se3,
        "semeval_3_grouped_gender_normalized_with_names.pdf",
        tight=False, plots_dir=plots_dir)

    plot_scaled_with_another_result(
        names_df_se3, gender_df_se3, names_names_se3, gender_names_se3,
        "semeval_3_names_gender_normalized_with_grouped_gender.pdf",
        tight=False, plots_dir=plots_dir)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--create-tests', action="store_true")
    parser.add_argument('--plots-dir', default=f"{cfg.ROOT}/plots")
    parser.add_argument('--classification',  action="store_true")
    parser.add_argument('--ner',  action="store_true")

    args = parser.parse_args()

    if args.create_tests:
        editor = Editor()
        fill_the_lexicon(editor)
        create_all_suites(editor)

    if not os.path.isdir(args.plots_dir):
        os.makedirs(args.plots_dir)

    if args.classification:
        get_results_for_model("roberta-semeval-2", attribute_names.keys())
        get_results_for_model("roberta-semeval-3", attribute_names.keys())

        plot_multiple_attributes(args.plots_dir)
        plot_gender_and_names(args.plots_dir)

    if args.ner:
        run_and_save_res(
            "grouped_country_by_gdp_ppp_quantile",
            "ner-roberta-conll2003", task="NER")

        load_and_plot(
            "grouped_country_by_gdp_ppp_quantile",
            "ner-roberta-conll2003",
            cl="LOC",
            plots_dir=args.plots_dir,
            skip_group_pcms=False,
            counterfactual=False)

        load_and_plot(
            "grouped_country_by_gdp_ppp_quantile",
            "ner-roberta-conll2003",
            cl="LOC",
            plots_dir=args.plots_dir,
            skip_group_pcms=False,
            counterfactual=True)

    if not args.classification and not args.ner:
        print("Provide at least one flag --classification or --ner")
