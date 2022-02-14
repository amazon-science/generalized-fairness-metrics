import logging
from overrides import overrides
from typing import Dict, Optional
import sys

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data import Tokenizer
from allennlp.data.tokenizers import Token
from allennlp.data.fields import TextField, LabelField, Field
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

from .sst_reader import StanfordSentimentTreeBankDatasetReaderPJC
from .semeval_reader import SemEvalReader
from .simple_reader import SimpleReader
from .huggingface_reader import HuggingfaceReader
from enum import Enum
logger = logging.getLogger(__name__)


class ReaderType(Enum):
    SST = 0
    PLAIN = 1
    HUGGINGFACE = 2
    SEMEVAL = 3
    ALL = 4


@DatasetReader.register("flexible_reader", exist_ok=True)
class FlexibleReader(DatasetReader):
    """
    A dataset reader wrapper.
    """
    def __init__(
        self,
        reader: str,
        token_indexers: Dict[str, TokenIndexer] = None,
        tokenizer: Optional[Tokenizer] = None,
        use_subtrees: bool = False,
        granularity: str = "5-class",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._tokenizer = tokenizer
        self._token_indexers =\
            token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.use_subtrees = use_subtrees
        self.granularity = granularity

        # cache to avoid creating many instances of the same reader
        self.readers_cache = {}

        self.reader = self.get_reader(reader)

    def get_reader(self, name: str):
        if name in self.readers_cache:
            return self.readers_cache[name]

        try:
            name = ReaderType[name]
        except KeyError:
            logger.error(f"Unsupported reader {name}")
            sys.exit()

        if name == ReaderType.SST:
            reader = StanfordSentimentTreeBankDatasetReaderPJC(
                token_indexers=self._token_indexers, tokenizer=self._tokenizer,
                use_subtrees=self.use_subtrees, granularity=self.granularity
            )
        elif name == ReaderType.PLAIN:
            reader = SimpleReader(
                token_indexers=self._token_indexers, tokenizer=self._tokenizer
            )
        elif name == ReaderType.HUGGINGFACE:
            reader = HuggingfaceReader(
                token_indexers=self._token_indexers, tokenizer=self._tokenizer
            )
        elif name == ReaderType.SEMEVAL:
            reader = SemEvalReader(
                token_indexers=self._token_indexers, tokenizer=self._tokenizer,
                granularity=self.granularity
            )
        elif name == ReaderType.ALL:
            reader = self
        else:
            logger.error(f"Unsupported reader {name}")
            sys.exit()

        self.readers_cache[name] = reader
        return reader

    @overrides
    def _read(self, file_path):
        if self.reader != self:
            for instance in self.reader._read(file_path):
                yield instance

        # NOTE: the following is not used at the moment.
        else:
            # this reader can adjust to parsing different types of examples on
            # the fly (e.g. one example is in the SST format, one in SemEval
            # etc.), but there are some conditions:
            # (i) there has to be one example per line in the file
            # (ii) each line has to be marked with what reader to use on it
            # (ii) the reader has to implement a _read_line function
            with open(cached_path(file_path), "r") as data_file:
                logger.info(
                    "Reading instances from lines in file at: %s", file_path)

                for i, line in enumerate(data_file.readlines()):
                    line = line.strip("\n")
                    reader, line = line.split("\t", 1)
                    if reader not in ["SST", "PLAIN", "SEMEVAL"]:
                        raise Exception(
                            f"Incorrect reader {reader} in line {i}")
                    if not line:
                        continue

                    reader = self.get_reader(reader)
                    for instance in reader._read_line(line):
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

        text_field = TextField(tokens, token_indexers=self._token_indexers)
        fields: Dict[str, Field] = {"tokens": text_field}

        if sentiment is not None:
            fields["label"] = LabelField(sentiment)
        return Instance(fields)
