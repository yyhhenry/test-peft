import logging
from typing import Any, NamedTuple
from datasets import load_dataset, DatasetDict, Dataset


class Sst2Dataset(NamedTuple):
    train: Dataset
    validation: Dataset
    test: Dataset


def load_sst2():
    dataset = load_dataset("glue", "sst2")
    assert isinstance(dataset, DatasetDict)
    logging.info(dataset)

    train = dataset["train"]
    assert isinstance(train, Dataset)
    validation = dataset["validation"]
    assert isinstance(validation, Dataset)
    test = dataset["test"]
    assert isinstance(test, Dataset)

    return Sst2Dataset(train=train, validation=validation, test=test)


class DataRow(NamedTuple):
    sentence: str
    label: int

    @classmethod
    def from_dict(cls, d: dict[str, Any]):
        return cls(sentence=d["sentence"], label=d["label"])
