from tqdm import tqdm
from util.dataset import DataRow, load_sst2
from util.log import init_logging, print_info
from util.model import get_peft, load_roberta
import torch

from util.train import train

init_logging()
torch.manual_seed(42)

sst2_dataset = load_sst2()
roberta = load_roberta()


def readable_test(title: str):
    print_info(f"{'v'*10} Readable Test | {title} {'v'*10}")
    label_name = {0: "0 Negative", 1: "1 Positive"}
    for index in range(10):
        row = DataRow.from_dict(sst2_dataset.validation[index])
        output_label = roberta.predict(row.sentence)
        print_info(f"Input: {row.sentence}")
        print_info(
            f"Output: {label_name[output_label]} / Answer: {label_name[row.label]} / Correct: {output_label == row.label}"
        )
    print_info(f"{'^'*10} Readable Test | {title} {'^'*10}")


def whole_test(title: str):
    print_info(f"{'v'*10} Whole Test | {title} {'v'*10}")
    n = len(sst2_dataset.validation)
    correct_count = 0
    for index in tqdm(range(n)):
        row = DataRow.from_dict(sst2_dataset.validation[index])
        output_label = roberta.predict(row.sentence)
        if output_label == row.label:
            correct_count += 1
    print_info(f"Accuracy: {correct_count}/{n} = {correct_count/n}")
    print_info(f"{'^'*10} Whole Test | {title} {'^'*10}")


readable_test("Before fine-tuning")
whole_test("Before fine-tuning")

peft = get_peft(roberta)
train(peft, roberta, sst2_dataset)

readable_test("After fine-tuning")
whole_test("After fine-tuning")
