from tqdm import tqdm
from transformers.modeling_outputs import SequenceClassifierOutput
from util.dataset import DataRow, load_sst2
from util.log import init_logging
from util.model import load_roberta
import torch

init_logging()
torch.manual_seed(42)

sst2_dataset = load_sst2()
roberta = load_roberta()


def readable_test(title: str):
    print(f"{'v'*10} Readable Test | {title} {'v'*10}")
    label_name = {0: "0 Negative", 1: "1 Positive"}
    for index in range(10):
        row = DataRow.from_dict(sst2_dataset.validation[index])
        token = roberta.tokenize(row.sentence)
        output = roberta.model(token)
        assert isinstance(output, SequenceClassifierOutput)
        output_label = int(output.logits.argmax().item())
        print(f"Input: {row.sentence}")
        print(
            f"Output: {label_name[output_label]} / Answer: {label_name[row.label]} / Correct: {output_label == row.label}"
        )
    print(f"{'^'*10} Readable Test | {title} {'^'*10}")


def whole_test(title: str):
    print(f"Whole Test | {title}:")
    n = len(sst2_dataset.validation)
    correct_count = 0
    for index in tqdm(range(n)):
        row = DataRow.from_dict(sst2_dataset.validation[index])
        token = roberta.tokenize(row.sentence)
        output = roberta.model(token)
        assert isinstance(output, SequenceClassifierOutput)
        output_label = int(output.logits.argmax().item())
        if output_label == row.label:
            correct_count += 1
    print(f"Accuracy: {correct_count}/{n} = {correct_count/n}")


readable_test("Before fine-tuning")
whole_test("Before fine-tuning")
