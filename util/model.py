from typing import NamedTuple
import logging
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput
from peft.tuners.lora import LoraConfig
from peft.mapping import get_peft_model
from peft.peft_model import PeftModel

CHECKPOINT = "roberta-base"


class Roberta(NamedTuple):
    model: RobertaForSequenceClassification
    tokenizer: RobertaTokenizer

    def tokenize(self, sentence: str):
        return self.tokenizer.encode(
            sentence, return_tensors="pt", truncation=True, max_length=512
        )

    def predict(self, sentence: str):
        token = self.tokenize(sentence)
        output = self.model(token)
        assert isinstance(output, SequenceClassifierOutput)
        output_label = int(output.logits.argmax().item())
        return output_label


def load_roberta():
    model = RobertaForSequenceClassification.from_pretrained(CHECKPOINT, num_labels=2)
    assert isinstance(model, RobertaForSequenceClassification)

    tokenizer = RobertaTokenizer.from_pretrained(CHECKPOINT)
    assert isinstance(tokenizer, RobertaTokenizer)

    logging.info(model)
    logging.info(tokenizer)
    return Roberta(model=model, tokenizer=tokenizer)


def get_peft(roberta: Roberta):
    peft_config = LoraConfig(
        task_type="SEQ_CLS",
        r=4,
        lora_alpha=32,
        lora_dropout=0.01,
        target_modules=["query"],
    )
    model = get_peft_model(roberta.model, peft_config)
    assert isinstance(model, PeftModel)
    model.print_trainable_parameters()
    logging.info(model)
    return model
