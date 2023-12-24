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
    logging.info(model)
    assert isinstance(model, RobertaForSequenceClassification)
    tokenizer = RobertaTokenizer.from_pretrained(CHECKPOINT)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        model.resize_token_embeddings(len(tokenizer))
    return RobertaModule(model=model, tokenizer=tokenizer)
