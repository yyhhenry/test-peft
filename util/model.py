from typing import NamedTuple
import logging
from transformers import RobertaForSequenceClassification, RobertaTokenizer

CHECKPOINT = "roberta-base"

# define label maps
id2label = {0: "Negative", 1: "Positive"}
label2id = {"Negative": 0, "Positive": 1}


class RobertaModule(NamedTuple):
    model: RobertaForSequenceClassification
    tokenizer: RobertaTokenizer


def load_roberta():
    model = RobertaForSequenceClassification.from_pretrained(
        CHECKPOINT, num_labels=2, id2label=id2label, label2id=label2id
    )
    assert isinstance(model, RobertaForSequenceClassification)

    tokenizer = RobertaTokenizer.from_pretrained(CHECKPOINT)
    assert isinstance(tokenizer, RobertaTokenizer)

    logging.info(model)
    logging.info(tokenizer)
    return RobertaModule(model=model, tokenizer=tokenizer)
