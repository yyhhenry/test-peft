from util.dataset import load_sst2
from util.model import get_peft, load_roberta, predict
from peft.peft_model import PeftModel

from util.test import readable_test, whole_test

sst2_dataset = load_sst2()
roberta = load_roberta()
peft = PeftModel.from_pretrained(
    roberta.model, "./model/roberta-text-classification-lora"
)
print("Model loaded.")

readable_test("After fine-tuning", peft, roberta.tokenizer, sst2_dataset.validation)
whole_test("After fine-tuning", peft, roberta.tokenizer, sst2_dataset.validation)
