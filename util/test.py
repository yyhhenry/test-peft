from tqdm import tqdm
from util.dataset import DataRow
from util.log import print_info
from util.model import predict
from peft.peft_model import PeftModel
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from datasets import Dataset


def readable_test(
    title: str,
    model: RobertaForSequenceClassification | PeftModel,
    tokenizer: RobertaTokenizer,
    validation: Dataset,
):
    print_info(f"{'v'*10} Readable Test | {title} {'v'*10}")
    label_name = {0: "0 Negative", 1: "1 Positive"}
    for index in range(10):
        row = DataRow.from_dict(validation[index])
        output_label = predict(model, tokenizer, row.sentence)
        print_info(f"Input: {row.sentence}")
        print_info(
            f"Output: {label_name[output_label]} / Answer: {label_name[row.label]} / Correct: {output_label == row.label}"
        )
    print_info(f"{'^'*10} Readable Test | {title} {'^'*10}")


def whole_test(
    title: str,
    model: RobertaForSequenceClassification | PeftModel,
    tokenizer: RobertaTokenizer,
    validation: Dataset,
):
    print_info(f"{'v'*10} Whole Test | {title} {'v'*10}")
    n = len(validation)
    correct_count = 0
    for index in tqdm(range(n)):
        row = DataRow.from_dict(validation[index])
        output_label = predict(model, tokenizer, row.sentence)
        if output_label == row.label:
            correct_count += 1
    print_info(f"Accuracy: {correct_count}/{n} = {correct_count/n}")
    print_info(f"{'^'*10} Whole Test | {title} {'^'*10}")
