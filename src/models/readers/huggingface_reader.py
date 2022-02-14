from typing import Dict, Optional
import logging

from allennlp.data import Tokenizer
from overrides import overrides

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, Field, LabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

from datasets import load_dataset
from sklearn.model_selection import train_test_split

import src.config as cfg

logger = logging.getLogger(__name__)


@DatasetReader.register("huggingface_reader", exist_ok=True)
class HuggingfaceReader(DatasetReader):
    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer] = None,
        tokenizer: Optional[Tokenizer] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._token_indexers =\
            token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._tokenizer = tokenizer

    @overrides
    def _read(self, file_path):
        # NOTE: file_path is not actually a file path but a name of the
        # dataset in huggingface datasets, followed by @, followed by the type
        # of split, e.g. "rotten_tomatoes@test" (a bit hacky but works well)
        dname, split = file_path.split("@")

        if split not in ["train", "test", "dev"]:
            raise Exception(f"Incorrect data split {split}")

        logger.info(
            f"Reading instances from the dataset: {file_path} ({split})")

        data = load_dataset(dname)
        # If the dataset doesn't have a separate dev split, get one from the
        # training data
        if "dev" not in data and split != "test":
            X_train, X_dev, y_train, y_dev = train_test_split(
                data['train']['text'],
                data['train']['label'],
                test_size=0.2, random_state=42)
            if split == "dev":
                sents = X_dev
                labels = y_dev
            else:
                sents = X_train
                labels = y_train
        else:
            sents = data[split]['text']
            labels = data[split]['label']

        for sent, label in zip(sents, labels):
            instance = self.text_to_instance(sent, str(label))
            if instance is not None:
                yield instance

    @overrides
    def text_to_instance(
        self,
        line: str,
        sentiment: str = None
    ) -> Optional[Instance]:
        if self._tokenizer is None:
            tokens = [Token(t) for t in line.split(" ")]
        else:
            tokens = self._tokenizer.tokenize(line)

        tokens = tokens[:cfg.max_position_embeddings]

        text_field = TextField(tokens, token_indexers=self._token_indexers)
        fields: Dict[str, Field] = {"tokens": text_field}

        if sentiment is not None:
            fields["label"] = LabelField(sentiment)
        return Instance(fields)
