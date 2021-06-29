from typing import Dict, Optional
import logging

from allennlp.data import Tokenizer
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, Field, LabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

from src.utils import join_tokens
import src.config as cfg

logger = logging.getLogger(__name__)


@DatasetReader.register("simple_reader", exist_ok=True)
class SimpleReader(DatasetReader):
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
        with open(cached_path(file_path), "r") as data_file:
            logger.info(
                "Reading instances from lines in file at: %s", file_path)

            for i, line in enumerate(data_file.readlines()):
                if not line:
                    continue
                for instance in self._read_line(line):
                    if instance is not None:
                        if i < 5:
                            logger.info(join_tokens(instance['tokens'].tokens))
                         #   print(join_tokens(instance['label'].label))
                        yield instance

    def _read_line(self, line):
        line = line.strip("\n")
        splits = line.split("\t")
        instance = None
        # NOTE and a TODO: for now this reader assumes there are no
        # tabs in the text itself
        if len(splits) == 2:
            # the line has the label and the text
            label, text = splits
            instance = self.text_to_instance(text, label)
        elif len(splits) == 3:
            # the line has the dataset name, the label and the text
            _, label, text = line.split("\t")
            instance = self.text_to_instance(text, label)
        elif len(splits) == 1:
            instance = self.text_to_instance(line)
        if instance is not None:
            yield instance

    @overrides
    def text_to_instance(
        self,
        line: str,
        label: str = None
    ) -> Optional[Instance]:
        if self._tokenizer is None:
            tokens = [Token(t) for t in line.split(" ")]
        else:
            tokens = self._tokenizer.tokenize(line)

        tokens = tokens[:cfg.max_position_embeddings]

        text_field = TextField(tokens, token_indexers=self._token_indexers)
        fields: Dict[str, Field] = {"tokens": text_field}

        if label is not None:
            fields["label"] = LabelField(label)
        return Instance(fields)
