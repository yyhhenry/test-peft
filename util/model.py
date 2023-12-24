from typing import NamedTuple
import logging
from torch import Tensor
from transformers import RobertaForSequenceClassification, RobertaTokenizer

CHECKPOINT = "roberta-base"


class Roberta(NamedTuple):
    model: RobertaForSequenceClassification
    tokenizer: RobertaTokenizer

    def tokenize(self, sentence: str):
        return self.tokenizer.encode(
            sentence, return_tensors="pt", truncation=True, max_length=512
        )


def load_roberta():
    model = RobertaForSequenceClassification.from_pretrained(CHECKPOINT, num_labels=2)
    assert isinstance(model, RobertaForSequenceClassification)

    tokenizer = RobertaTokenizer.from_pretrained(CHECKPOINT)
    assert isinstance(tokenizer, RobertaTokenizer)

    logging.info(model)
    logging.info(tokenizer)
    return Roberta(model=model, tokenizer=tokenizer)
