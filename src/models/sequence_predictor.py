from overrides import overrides
from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors.sentence_tagger import SentenceTaggerPredictor
from allennlp.predictors.predictor import Predictor
from allennlp.data.tokenizers import Token


@Predictor.register("sentence_tagger_pjc")
class SentenceTaggerPredictorPjc(SentenceTaggerPredictor):
    def __init__(
        self, model: Model, dataset_reader: DatasetReader,
        language: str = "en_core_web_sm"
    ) -> None:
        super().__init__(model, dataset_reader, language)

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like `{"tokenized_sentence": "..."}`.
        Runs the underlying model, and adds the `"words"` to the output.
        """
        tokens = json_dict["tokenized_sentence"]
        tokens = [Token(t) for t in tokens]
        return self._dataset_reader.text_to_instance(tokens)
