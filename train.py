from util.dataset import load_sst2
from util.log import init_logging, print_info
from util.model import get_peft, load_roberta
from util.test import readable_test, whole_test
from util.train import train
import torch

init_logging("train")
torch.manual_seed(42)

sst2_dataset = load_sst2()
roberta = load_roberta()


readable_test(
    "Before fine-tuning", roberta.model, roberta.tokenizer, sst2_dataset.validation
)
whole_test(
    "Before fine-tuning", roberta.model, roberta.tokenizer, sst2_dataset.validation
)

peft = get_peft(roberta)
train(peft, roberta, sst2_dataset)

peft.save_pretrained("model/roberta-text-classification-lora")
print_info("Model saved.")
