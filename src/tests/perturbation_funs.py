from checklist_fork.checklist.editor import Editor
from checklist_fork.checklist.utils import get_class_from_seq_label, tokenize

from typing import List, Callable, Union, Tuple, Collection
import numpy as np
import regex as re
from collections import defaultdict, namedtuple
from spacy.symbols import nsubj, VERB, AUX
import spacy
from src.tests.fill_the_lexicon import read_matched_terms

import src.config as cfg
import logging
from munch import Munch

logging.basicConfig(level=cfg.log_level)
logger = logging.getLogger(__name__)

# TODO: read those from file
tup_attributes = ["male", "female", "neut"]
Tup = namedtuple("Tup", tup_attributes)


replacement_bundles = []
paired_gendered_terms: List[Munch] =\
    read_matched_terms(
        "matched_gendered_terms", constraints={"GROUP": "family"})

for pair in paired_gendered_terms:
    tup1 = Tup(pair["MALE"], pair["FEMALE"], pair["NEUTRAL"])
    tup1 = Tup(
        pair["MALE"].capitalize(), pair["FEMALE"].capitalize(),
        pair["NEUTRAL"].capitalize())

nlp = spacy.load('en_core_web_sm')

# replacement_bundles = [
#     Tup("mr.", "ms.", "mx."),
#     Tup("mr", "ms", "mx"),
#     Tup("actor", "actress", "actor"),
#     Tup('himself', 'herself', 'themselves'),
#     Tup('man', 'woman', 'person'),
#     Tup('husband', 'wife', 'life partner'),
#     Tup('boy', 'girl', 'youth'),
#     Tup('boyfriend', 'girlfriend', 'partner'),
#     Tup('he', 'she', 'they'),
#     Tup("he's", "she's", "they're"),
#     Tup('him', 'her', 'them'),
#     Tup('his', 'her', 'their'),
#     Tup('male', 'female', 'non-binary'),
#     # Tup('gentlemen', 'ladies', 'gentlepeople'),
#     Tup('gentlemen', 'lady', 'person')
# ]

# for i in range(len(replacement_bundles)):
#     tup = replacement_bundles[i]
#     replacement_bundles.append(
#         Tup(tup.male.capitalize(), tup.female.capitalize(),
#             tup.neut.capitalize())
#     )


##################################
#   PERTURBATION FUNCTIONS       #
##################################

def add_phrases(
    phrases: List[str]
) -> Callable[[str], List[str]]:
    def pert(d):
        ret = [d + " " + x for x in phrases]
        idx = np.random.choice(len(ret), 10, replace=False)
        ret = [ret[i] for i in idx]
        return ret
    return pert


def sub_helper(toks_ws, index, new_word):
    """
    A helper for word substitutions, changes word x to y in the given slot of
    the toks_ws list which contains the word x alogn with some whitespaces.
    E.g.
        toks_ws = ["This ", "is ", "an ", "example", "."]
        index = 1
        new_word = "likes"
        => ["This", "likes ", "an ", "example", "."]
    """
    toks_ws[index] = re.sub(toks_ws[index].strip(), new_word, toks_ws[index])


##################################
#   HELPERS FOR HANGING GENDER   #
##################################

def reinflect_verbs_headed_by_prons(
    doc: List,
    neutral_tokens: List
) -> None:
    """
    This function is used to reinflect verbs which are headed by pronoun
    he/she and that pronoun turns to 'they'.
    """
    verbs_heading_prons = []
    for x in doc:
        if x.text.lower() in ['he', 'she']:
            if x.dep == nsubj and x.head.pos in [VERB, AUX]:
                verbs_heading_prons.append(x.head)
    for x in doc:
        if x.pos == AUX and x.head in verbs_heading_prons:
            # in such cases it is the AUX that agrees with the subj
            verbs_heading_prons.remove(x.head)
            verbs_heading_prons.append(x)

    for v in verbs_heading_prons:
        target = None
        if v.text == "was":
            target = "were"
        elif v.text == "is":
            target = "are"
        elif v.text == "'s":
            v.text = "'re"
        elif v.tag_ == "VBZ":
            target = v.lemma_
        if target:
            assert neutral_tokens[v.i] == v.text_with_ws
            # keep the original spaces
            sub_helper(neutral_tokens, v.i, target)


def change_her(
    doc: List,
    perturbed_tokens: List,
    possesive: str,
    accusative: str,
) -> None:
    """
    Handle changing her to his/their or him/them (dependent on the POS)
    """
    for x in doc:
        if x.text == "her" or x.text == "Her":
            caps = (x.text == "Her")
            if x.tag_ == "PRP$":
                sub_helper(perturbed_tokens, x.i,
                           f"xxx{possesive.lower()}xxx" if not caps
                           else f"xxx{possesive.capitalize()}xxx")
            elif x.tag_ == "PRP":
                sub_helper(perturbed_tokens, x.i,
                           f"xxx{accusative.lower()}xxx" if not caps
                           else f"xxx{accusative.capitalize()}xxx")


def change_terms(
    perturbed_tokens: str,
    # possible att are: ["female", "male", "neut"]
    changed_target_att_bundles: List[Tuple[str, str]]
):
    for tup in replacement_bundles:
        for changed_att, target_att in changed_target_att_bundles:
            try:
                target = getattr(tup, target_att)
                to_change = getattr(tup, changed_att)
                perturbed_tokens = [re.sub(
                    r'\b%s\b' % re.escape(to_change),
                    f'xxx{re.escape(target)}xxx', x) for x in perturbed_tokens]
            except AttributeError:
                raise Exception(
                    f"Not recognized attribute: {changed_att} or {target_att}")
    return perturbed_tokens


###############################
#   CORE PERTURBATION FUNS    #
###############################

def change_genders(
    doc,
    meta: bool = False,
    target_gender: str = "female",  # ["female", "male", "swap", "neut"]
) -> Union[Tuple[List, List], List]:
    """
    This changes the pronouns and some gendered terms, but doesn't change the
    names (!). It can be used to evaluate how the model reacts to mismatches
    between the typical gender associated with the name and the gendered terms
    that accompany it.

    The perturbation returns a version of the sentence in which all gendered
    terms have been changed to manifest the target_gender. If target_gender is
    swap it returns a sentence in which all male terms were turned to female
    terms and female to male terms.
    """

    # operating on tokens allows for replacing specific instances of words
    # rather than all words that match a particular form. We use text_with_ws
    # to allow for recreating the original string from the tokens.
    perturbed_toks = [x.text_with_ws for x in doc]

    # token specific changes (dependent on syntax)
    if target_gender == "neut":
        reinflect_verbs_headed_by_prons(doc, perturbed_toks)
        change_her(doc, perturbed_toks, 'their', 'them')
    elif target_gender in ["swap" or "male"]:
        change_her(doc, perturbed_toks, 'his', 'him')

    # replace all gendered terms, xxx padding is added for the once replaced
    # words to stay as they are
    if target_gender == "swap":
        perturbed_toks = change_terms(
            perturbed_toks, [("male", "female"), ("female", "male")])
    else:
        for att in tup_attributes:
            if att != target_gender:
                perturbed_toks = change_terms(
                    perturbed_toks, [(att, target_gender)])

    # clean the xxx padding
    perturbed_toks = [
        re.sub(r'xxx(.+?)xxx', r'\g<1>', x) for x in perturbed_toks]

    ret = []
    ret_m = []

    perturbed_sent = ''.join(perturbed_toks)
    if perturbed_sent != doc.text:
        ret.append((perturbed_sent, perturbed_toks))
        ret_m.append(target_gender)
    return (ret, ret_m) if meta else ret


# NOTE: NOT USED ATM, but might be used in the future
# def full_change_genders_and_names(
#     doc,
#     male_names: Collection,
#     female_names: Collection,
#     meta: bool = False,
#     seed: int = None,
#     n: int = 20
# ):
#     """
#     Change in a sentence to all female/all male/swap male to female
#     """
#     ret, ret_m = [], []
#     for gend in ["female", "male", "swap"]:
#         tmp_ret, tmp_meta = change_genders_and_names(
#             doc, male_names, female_names, meta=True, seed=seed, n=n,
#             target_gender=gend, terms_gender=gend)
#         if not tmp_ret:
#             continue
#         for s, m in zip(tmp_ret, tmp_meta):
#             # e.g. if a sentence has one fem name then all male and swap
#             # would be the same
#             if s not in ret:
#                 ret.append(s)
#                 ret_m.append(m)
#     return (ret, ret_m) if meta else ret


def change_genders_and_names(
    doc,
    male_names: Collection,
    female_names: Collection,
    names_to_change: Collection,
    meta: bool = False,
    seed: int = None,
    n: int = 20,
    target_gender: str = "female",  # ["female", "male", "swap"]
    terms_gender: str = "female"  # ["female", "male", ""swap", "neut"]
    # source_gender: str = "either",  # male, female
) -> Union[Tuple[List, List], List]:
    """
    This the first names and the gendered terms and pronouns.
    (NOTE: if a sentence didn't contain a name swap then it's not returned!)

    target_gender argument controls how the names are changed.
    If swap is used then names that are typically associated with
    males are turned to names typically associated with females and vice-versa.
    Names unrecognized as typically male or typically female are left as they
    are.
    If female -- all names recognised as typically male are changed to female
    names and all other names remain unchanged.
    If male -- analogous to female

    terms_gender controls how the gendered terms are changed. This should
    typically be the same as target_gender or "neut".
    """
    if seed is not None:
        np.random.seed(seed)

    # NER perturbations; one can provide both parsed sentence as well as ents
    if type(doc) == tuple:
        doc, entities = doc
        assert len(doc) == len(entities)
        # todo
        ents = [x.text for x, e in zip(doc, entities)
                if get_class_from_seq_label(e) == "PER"]
    else:
        ents = [x.text for x in doc.ents if np.all(
                [a.ent_type_ == 'PERSON' for a in x])]


    # recording choices for each name and replacing later ensures that
    # all occurences of the name are replaced to the same new name
    fname2selected = defaultdict(list)
    for ent in ents:
        fname = ent.split()[0]
        if fname in ["Oscar", "Jesus"]:
            continue

        if fname not in names_to_change:
            continue

        names = female_names if target_gender == "female" else male_names

        # ensure we don't sample the original name
        names = set(names)
        names.discard(fname)
        names = list(names)
        selected = np.random.choice(names, n, replace=False)
        fname2selected[fname] = selected

    ret = []
    ret_m = []

    if len(fname2selected) == 0:
        return (ret, ret_m) if meta else ret

    try:
        _, perturbed_toks = change_genders(
            doc, target_gender=terms_gender)[0]
    except IndexError:
        perturbed_toks = [x.text_with_ws for x in doc]

    for i in range(n):
        toks = perturbed_toks
        info = []

        for fname, selected in fname2selected.items():
            y = selected[i]
            info.append((fname, y))

            # NOTE: This assumes that the name is tokenized in the same way out
            # of context and in the given context
            name_tokens = [
                x.text_with_ws for x in list(nlp.pipe([str(y)], disable=[
                    "tagger", "parser", "ner"]))[0]]

            new_toks = []
            for t in toks:
                m = re.search(r'(.*)\b%s\b(.*)' % re.escape(fname), t)

                if m:
                    # add whitespaces
                    name_tokens[0] = m.group(1) + name_tokens[0]
                    name_tokens[-1] = name_tokens[-1] + m.group(2)
                    new_toks += name_tokens
                else:
                    new_toks.append(t)
            toks = new_toks
            # toks = [
            #     re.sub(r'\b%s\b' % re.escape(fname), y, x)
            #     for x in toks]

            # replace only one name
            break

        sent = ''.join(toks)
        if sent != doc.text:
            ret.append((sent, toks))
            ret_m.append(Munch({
                "info": info,
                "names_target": target_gender,
                "terms_target": terms_gender
            }))
    return (ret, ret_m) if meta else ret


def change_nationalities(
    doc,
    editor: Editor,
    target_nationalities: Collection,
    meta: bool = False,
    seed: int = None,
    n: int = 20
) -> Union[Tuple[List, List], List]:
    if seed is not None:
        np.random.seed(seed)
    ents = [x.text for x in doc.ents if
            np.all([a.ent_type_ == 'NORP' for a in x])]

    # extra filter -- spacy tools are not perfect
    ents = [x for x in ents if
            x.capitalize() in editor.lexicons['nationality']]

    ret = []
    ret_m = []
    for ent in ents:
        select_set = set(target_nationalities)
        select_set.discard(ent)
        select_set = list(select_set)
        selected = np.random.choice(select_set, n, replace=False)
        for y in selected:
            ret.append(re.sub(r'\b%s\b' % re.escape(ent), y, doc.text))
            ret_m.append((ent, y))
    return (ret, ret_m) if meta else ret
