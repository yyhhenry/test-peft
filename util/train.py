import logging
from typing import Any
from peft.peft_model import PeftModel
from torch import Tensor
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from util.dataset import DataRow, Sst2Dataset
from util.model import CHECKPOINT, Roberta


def train(peft: PeftModel, roberta: Roberta, dataset: Sst2Dataset):
    lr = 1e-3
    batch_size = 16
    num_epochs = 5

    def dataset_map_fn(example: dict[Any, Any]):
        row = DataRow.from_dict(example)
        token = roberta.tokenize(row.sentence)
        return {"sentence": token, "label": row.label}

    train = dataset.train.map(dataset_map_fn)
    validation = dataset.validation.map(dataset_map_fn)
    logging.info("Converted to tokenized dataset")

    data_collator = DataCollatorWithPadding(tokenizer=roberta.tokenizer)

    args = TrainingArguments(
        output_dir="model/roberta-text-classification-lora",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    def compute_metrics(p: tuple[Tensor, Tensor]):
        predictions, labels = p
        predictions = predictions.argmax(axis=1)
        accuracy = Tensor(predictions - labels).square().mean()
        return {"accuracy": accuracy}

    trainer = Trainer(
        model=peft,
        args=args,
        train_dataset=train,
        eval_dataset=validation,
        tokenizer=roberta.tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
