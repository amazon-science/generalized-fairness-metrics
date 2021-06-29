from typing import List, Callable, Dict
from checklist_fork.checklist.editor import Editor

import spacy
from spacy.tokens import Doc

from munch import Munch


def split_dataset(
    data: List[Doc],
    labels: List,
    bucket_funs: Dict[str, Callable[[Doc], bool]],
    only_one_bucket: bool = True
) -> Dict[str, Munch]:
    splits: Dict[str, Munch] =\
        {name: Munch({"data": [], "labels": []})
         for name in bucket_funs.keys()}

    for ex, lab in zip(data, labels):
        for name, bfun in bucket_funs.items():
            if bfun(ex):
                splits[name].data.append(ex)
                splits[name].labels.append(lab)
                if only_one_bucket:
                    continue
    return splits


def get_female_male_name_splits(
    editor: Editor,
    data: List[str],
    labels: List
) -> Dict[str, Munch]:

    nlp = spacy.load('en_core_web_sm')
    data = list(nlp.pipe(data))

    male_names = editor.lexicons['male']
    female_names = editor.lexicons['female']

    def male_fun(ex: Doc):
        return any([x.text in male_names for x in ex]) and \
            all([x.text not in female_names for x in ex])

    def female_fun(ex: Doc):
        return any([x.text in female_names for x in ex]) and \
            all([x.text not in male_names for x in ex])

    return split_dataset(
        data, labels,
        {"male": male_fun, "female": female_fun},
        only_one_bucket=True)
