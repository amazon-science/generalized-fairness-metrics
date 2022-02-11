
from src.models.readers import SemEvalReader, HuggingfaceReader, \
    StanfordSentimentTreeBankDatasetReaderPJC

from allennlp.data.dataset_readers.conll2003 import Conll2003DatasetReader

from typing import List, Tuple

import src.config as cfg
from src.utils import join_tokens


def read_data(
    dataset: str,
    split: str  # [train, test, dev]
) -> Tuple[List[str], List[int]]:
    """
    Function that provides an abstraction over the format of the data
    in the given dataset and returns a list of examples and a list of labels.
    """
    supported_datasets = [
        "sst-2", "sst-3", "semeval-2", "semeval-3", "conll2003"
    ]
    if dataset not in supported_datasets:
        raise Exception(f"Dataset {dataset} not supported.")
    if split not in ["train", "test", "dev"]:
        raise Exception(f"Split {split} not supported.")

    pretokenized = False
    if "sst" in dataset:
        granularity = f"{dataset.split('-', 1)[1]}-class"
        reader =\
            StanfordSentimentTreeBankDatasetReaderPJC(
                use_subtrees=False, granularity=granularity)
        splits = {"dev": cfg.dev_sst_path,
                  "test": cfg.test_sst_path, "train": cfg.train_sst_path}
        pretokenized = True

    elif "semeval" in dataset:
        granularity = f"{dataset.split('-', 1)[1]}-class"
        reader = SemEvalReader(granularity=granularity)
        splits = {"dev": cfg.dev_semeval_path,
                  "test": cfg.test_semeval_path,
                  "train": cfg.train_semeval_path}
    elif "conll2003" in dataset:
        reader = Conll2003DatasetReader(coding_scheme="BIOUL")
        splits = {"dev": cfg.dev_conll_path,
                  "test": cfg.test_conll_path,
                  "train": cfg.train_conll_path}
    else:
        reader = HuggingfaceReader()
        splits = {"dev": f"{dataset}@dev",
                  "test":  f"{dataset}@test", "train": f"{dataset}@train"}

    fpath = splits[split]
    labels = []
    sentences = []
    all_tokens = []
    gen = reader._read(fpath)
    for instance in gen:
        tokens = instance['tokens'].tokens
        tokens = [t.text for t in tokens]

        if 'label' in instance:
            label = instance['label'].label
            labels.append(int(label))
        else:
            label = instance['tags'].labels
            labels.append(label)

        # clean argument determines if spaces around certain chars are removed
        # e.g. "This is a sentence ." -> "This is a sentence."
        text = join_tokens(tokens, clean=pretokenized)
        sentences.append(text)
        all_tokens.append(tokens)
    return sentences, labels, all_tokens
