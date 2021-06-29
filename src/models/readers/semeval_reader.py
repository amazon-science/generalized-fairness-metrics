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

logger = logging.getLogger(__name__)

import src.config as cfg

@DatasetReader.register("semeval_reader", exist_ok=True)
class SemEvalReader(DatasetReader):
    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer] = None,
        tokenizer: Optional[Tokenizer] = None,
        granularity: str = "2-class",  # 3-class, 7-class
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._token_indexers =\
            token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._tokenizer = tokenizer
        self._granularity = granularity

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info(
                "Reading instances from lines in file at: %s", file_path)

            for line in data_file.readlines():
                if not line:
                    continue
                for instance in self._read_line(line):
                    if instance is not None:
                        yield instance

    def _read_line(self, line):
        line = line.strip("\n")
        parts = line.split("\t")
        if len(parts) != 4 or parts[0] == "ID":
            yield None
        else:
            label = parts[-1].split(":", 1)[0]
            text = parts[1]
            instance = self.text_to_instance(text, label)
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

        # as in SST allow for different number of classes. For 2 and 3 classes
        # the scale is remapped: instead of negative integers
        # (original [-3, 3]) use [0,1] or [0,2]
        if label is not None:
            if self._granularity == "3-class":
                if int(label) < 0:
                    label = "0"
                elif int(label) == 0:
                    label = "1"
                else:
                    label = "2"
            elif self._granularity == "2-class":
                if int(label) < 0:
                    label = "0"
                # an in SST-2 remove neutral sentiment
                elif int(label) == 0:
                    return None
                else:
                    label = "1"
            fields["label"] = LabelField(label)
        return Instance(fields)
