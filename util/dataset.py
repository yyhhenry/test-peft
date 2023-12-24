import logging
from typing import NamedTuple
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
