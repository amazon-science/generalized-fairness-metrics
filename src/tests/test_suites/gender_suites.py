import sys
sys.path.insert(0, ".")

import pandas as pd
import src.config as cfg
import src.tests.process_templates as tp
import src.tests.test_creation_utils as test_utils
from src.tests.save_and_load import save_suite
from checklist_fork.checklist.editor import Editor
from checklist_fork.checklist.test_suite import TestSuite
from checklist_fork.checklist.perturb import Perturb

from typing import Dict, List
from checklist_fork.checklist.eval_core import EvaluationCore
from src.tests.fill_the_lexicon import read_matched_terms, fill_the_lexicon

from .explicit_terms_suites import update_lab_dict

from munch import Munch
import regex as re
from collections import defaultdict
import src.tests.perturbation_funs as pfuns
from functools import partial
import math
from src.models.readers.read_data import read_data
import spacy
from spacy.tokens import Doc


def create_names_evaluation_suite(
    editor: Editor,
    name_groups: Dict = {},
    name: str = 'male-vs-female-names',
    nsamples: int = 100,
    tpath: str = None,
    domain: str = None
):
    """
    If domain is given then the core will be created only on the sentences
    from that domain. But note that even if the core is created on all
    sentences -- during test time one can filter the sentences used for evalu-
    ation based on domain.
    """
    suite = TestSuite()

    if not tpath:
        tpath = cfg.main_template_path

    # defaults to male vs female (English names)
    if not name_groups:
        name_groups = {
            "male": editor.lexicons['male'],
            "female": editor.lexicons['female']
        }

    # label dict maps tasks to labels
    data, meta, labels_dict = [], [], defaultdict(Munch)
    for ik in ["person", "first_name"]:
        # get templates that are suitable to be filled with names
        df_templates: pd.DataFrame = tp.get_templates(
            editor, tpath, identity_keys={ik})

        if domain:
            df_templates = tp.get_df_for_domain(df_templates, domain)

        tmp_data, tmp_meta, tmp_labels_dict =\
            test_utils.get_grouped_data_from_template(
                editor=editor,
                df_templates=df_templates,
                nsamples=nsamples,
                identity_key=ik,
                groups=name_groups
            )

        data += tmp_data
        meta += tmp_meta
        update_lab_dict(labels_dict, tmp_labels_dict)

    ct = EvaluationCore(
        data, meta=meta, labels_dict=labels_dict,
        name=f"{name}", capability="Gender")
    suite.add(ct)

    save_suite(suite, name)
    return suite, name


# semantically grouped terms
def create_gendered_terms_evaluation_suite(
    editor: Editor,
    name: str = 'male-vs-female-paired-terms',
    nsamples: int = None
):
    suite = TestSuite()

    df_templates_person: pd.DataFrame = tp.get_templates(
        editor, cfg.main_template_path, identity_keys={'person'})

    # each bundle of paired matched terms should be a munch with keys
    # term1, term2 etc.
    paired_female_male_terms: List[Munch] =\
        read_matched_terms(
            "matched_gendered_terms",
            constraints={"GROUP": "family"},
            keys_to_skip={"GROUP", "NEUTRAL"})

    for x in paired_female_male_terms:
        for k, v in x.items():
            x[k] = f"my {v}"

    keys = paired_female_male_terms[0].keys()
    new_keys = [f"person.{key}" for key in keys]

    # label dict maps tasks to labels
    data, meta, labels_dict = test_utils.get_grouped_data_from_template(
        editor, df_templates_person,
        groups={"person": paired_female_male_terms},
        new_keys=new_keys,
        identity_key='person',
        nsamples=nsamples,
    )

    ct = EvaluationCore(
        data, meta=meta, labels_dict=labels_dict,
        name=f"{name}", capability="Gender")

    suite.add(ct)
    save_suite(suite, name)
    return suite, name


def create_ungrouped_gender_adj_suite(
    editor: Editor,
    name: str = 'ungrouped_gender_adjs'
):
    suite = TestSuite()

    df_templates = tp.get_templates(
        editor, cfg.main_template_path, identity_keys={'identity_adj'})

    # each group is represented by a single term
    groups = {}
    for adj in editor.lexicons['gender_adj_basic']:
        groups[adj] = adj

    # label dict maps tasks to labels
    data, meta, labels_dict = test_utils.get_grouped_data_from_template(
        editor=editor,
        df_templates=df_templates,
        nsamples=None,
        identity_key='identity_adj',
        groups=groups,
    )

    ct = EvaluationCore(
        data, meta=meta, labels_dict=labels_dict,
        name=f"{name}", capability="Gender")
    suite.add(ct)

    save_suite(suite, name)
    return suite, name


def create_ungrouped_gender_np_suite(
    editor: Editor,
    name: str = 'ungrouped_gender_nps'
):
    suite = TestSuite()

    df_templates = tp.get_templates(
        editor, cfg.main_template_path, identity_keys={'identity_np'})

    # each group is represented by a single term
    # get determiners for each term
    groups = {}
    for np in editor.lexicons['gender_n_basic']:
        groups[np] = editor.template('{a:x}', x=[np]).data[0]

    # one can also use the adjs to create 'x preson' nps
    # for np in editor.lexicons['gender_adj_basic']:
    #     groups[np] = editor.template('{a:x} person', x=[np]).data[0]

    data, meta, labels_dict = test_utils.get_grouped_data_from_template(
        editor=editor,
        df_templates=df_templates,
        nsamples=None,
        identity_key='identity_np',
        groups=groups
    )

    ct = EvaluationCore(
        data, meta=meta, labels_dict=labels_dict,
        name=f"{name}", capability="Gender")
    suite.add(ct)

    save_suite(suite, name)
    return suite, name


# text classification
def perturbation_name_suite(
    editor: Editor,
    name: str = 'perturbation_names',
    nsamples: int = 20
):
    suite = TestSuite()

    def create_core_from_perturb(
        parsed_data,
        all_labels,
        data_name,
        # that determines how the labels are added check config file for
        # supported tasks
        task,
        n_classes=None
    ):
        # we use the same tokens as the sources of perturbation for both
        # male versions of the sentences and female versions of the sentences
        # so we get exactly the same set of sentences from both perturbations
        n = nsamples

        all_names = set(editor.lexicons["all_female_names"] +
                        editor.lexicons["all_male_names"])

        pert_fun = partial(
            pfuns.change_genders_and_names,
            male_names=editor.lexicons["male"],
            female_names=editor.lexicons["female"],
            target_gender="male",
            terms_gender="male",
            names_to_change=all_names,
            n=n,
        )
        male_ret, male_labels = Perturb.perturb(
            parsed_data, pert_fun, nsamples=None, meta=True, labels=all_labels)

        # female sents
        pert_fun = partial(
            pfuns.change_genders_and_names,
            male_names=editor.lexicons["male"],
            female_names=editor.lexicons["female"],
            target_gender="female",
            terms_gender="female",
            names_to_change=all_names,
            n=n,
        )
        fem_ret, fem_labels = Perturb.perturb(
            parsed_data, pert_fun, nsamples=None, meta=True, labels=all_labels)

        # same sentences = same labels
        assert fem_labels == male_labels

        data = []
        meta = []
        labels = []
        for msents, fsents, mmeta, fmeta, lab in zip(
                male_ret.data, fem_ret.data, male_ret.meta,
                fem_ret.meta, male_labels):

            # the first sentence is the original sentence
            assert msents[0] == fsents[0]
            template = msents[0]

            # info in this is a list of a single tuple with original name and
            # new name, e.g. [('Johnny', 'Kelly')], meta for the firs sent is
            # None
            identity_key = mmeta[1].info[0][0]

            # all sentences for both genders should be replacing the same
            # name in the original sentence
            assert all(met.info[0][0] == identity_key for met in mmeta[1:])
            assert all(met.info[0][0] == identity_key for met in fmeta[1:])

            male_fills = [met.info[0][1] for met in mmeta[1:]]
            female_fills = [met.info[0][1] for met in fmeta[1:]]

            example_meta = {
                    0: "male", 1: "female",
                    "IDENTITY_KEY": identity_key,
                    "SAMPLE": {"male": male_fills,
                               "female": female_fills},
                    "TEMPLATE": template
                }
            # many sentences for male group and many for female group
            # if necessary this structure is 'flattened' when the metric
            # is computed (e.g. through averaging or random pairing)
            example_data = [[sent for sent, toks in msents[1:]],
                            [sent for sent, toks in fsents[1:]]]

            data.append(example_data)
            meta.append(example_meta)

            # labels is shared for all instances of that sent.
            labels.append(lab)

        labels_dict = {task: Munch({"labels": labels, "n_classes": n_classes})}
        return EvaluationCore(data, meta=meta, labels_dict=labels_dict,
                              name=data_name, capability="Gender")

    test_utils.add_cores_from_perturbation(
        create_core_function=partial(create_core_from_perturb, n_classes=3),
        dataset='sst-3',
        suite=suite,
    )

    test_utils.add_cores_from_perturbation(
        create_core_function=partial(create_core_from_perturb, n_classes=3),
        dataset='semeval-3',
        suite=suite
    )
    save_suite(suite, name)
    return suite, name, None


def perturbation_conll_name_suite(
    editor: Editor,
    name: str = 'perturbation_conll_names',
    nsamples: int = 10
):
    dev_sents, dev_labels, dev_tokens = read_data("conll2003", "dev")

    nlp = spacy.load('en_core_web_sm')

    # force spacy to use the tokens from CoNLL data
    tokens_dict = {x: y for x, y in zip(dev_sents, dev_tokens)}

    def custom_tokenizer(text):
        if text in tokens_dict:
            toks = tokens_dict[text]
            d = Doc(nlp.vocab, toks)
            return d
        else:
            raise ValueError('No tokenization available for input.')

    nlp.tokenizer = custom_tokenizer
    parsed_data = list(nlp.pipe(dev_sents))
    parsed_data_and_entities = list(zip(parsed_data, dev_labels))

    # we use the same tokens as the sources of perturbation for both
    # male versions of the sentences and female versions of the sentences
    # so we get exactly the same set of sentences from both perturbations
    n = nsamples
    all_names = set(editor.lexicons["all_female_names"] +
                    editor.lexicons["all_male_names"])

    pert_fun = partial(
        pfuns.change_genders_and_names,
        male_names=editor.lexicons["male"],
        female_names=editor.lexicons["female"],
        target_gender="male",
        terms_gender="male",
        names_to_change=all_names,
        n=n
    )
    male_ret, male_labels = Perturb.perturb(
        parsed_data_and_entities,
        pert_fun, nsamples=None, meta=True,
        labels=dev_labels)

    # female sents
    pert_fun = partial(
        pfuns.change_genders_and_names,
        male_names=editor.lexicons["male"],
        female_names=editor.lexicons["female"],
        target_gender="female",
        terms_gender="female",
        names_to_change=all_names,
        n=n,
    )
    fem_ret, fem_labels = Perturb.perturb(
        parsed_data_and_entities,
        pert_fun, nsamples=None, meta=True, labels=dev_labels)

    # same sentences = same labels
    assert fem_labels == male_labels

    data = []
    meta = []
    labels = []
    for msents, fsents, mmeta, fmeta, lab in zip(
            male_ret.data, fem_ret.data, male_ret.meta,
            fem_ret.meta, male_labels):

        # the first sentence is the original sentence
        assert msents[0] == fsents[0]

        # template is the sentence and the entities (because we passed
        # parsed_data_and_entities to the perturb fun)
        template = msents[0][0]

        # info in this is a list of a single tuple with original name and
        # new name, e.g. [('Johnny', 'Kelly')], meta for the firs sent is
        # None
        identity_key = mmeta[1].info[0][0]

        # all sentences for both genders should be replacing the same
        # name in the original sentence
        assert all(met.info[0][0] == identity_key for met in mmeta[1:])
        assert all(met.info[0][0] == identity_key for met in fmeta[1:])

        male_fills = [met.info[0][1] for met in mmeta[1:]]
        female_fills = [met.info[0][1] for met in fmeta[1:]]

        male_data = []
        for sent, toks in msents[1:]:
            # toks include white spaces
            tokens_dict[sent] = [t.strip() for t in toks]
            male_data.append(sent)

        female_data = []
        for sent, toks in fsents[1:]:
            # toks include white spaces
            tokens_dict[sent] = [t.strip() for t in toks]
            female_data.append(sent)

        # many sentences for male group and many for female group
        # if necessary this structure is 'flattened' when the metric
        # is computed (e.g. through averaging or random pairing)
        example_data = [male_data, female_data]

        tokenized_template = custom_tokenizer(template.strip())
        tokenized_template = [
            t.text if t.text != identity_key
            else f"@{identity_key}@" for t in tokenized_template]
        example_meta = {
                0: "male", 1: "female",
                "IDENTITY_KEY": identity_key,
                "SAMPLE": {"male": male_fills, "female": female_fills},
                # the following tokenization fields are important to provide
                # for sequence labeling perturbation data
                # they ensure that the model's predictions match
                # the tokens assumed by the labels and allow for mismatches
                # in tokens for identity terms
                "TOKENIZATION_DICT": tokens_dict,
                "TOKENIZED_TEMPLATE": tokenized_template,
                "TEMPLATE": template
            }

        data.append(example_data)
        meta.append(example_meta)

        # labels is shared for all instances of that sent.
        labels.append(lab)

    labels_dict = {"NER": Munch({"labels": labels, "n_classes": None})}
    dev_core = EvaluationCore(
        data, meta=meta, labels_dict=labels_dict,
        name="Dev CoNLL 2003", capability="Gender")
    suite = TestSuite()
    suite.add(dev_core)
    save_suite(suite, name)

    return suite, name, data
