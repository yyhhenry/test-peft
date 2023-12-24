import torch
import numpy as np
import logging
from util.dataset import load_sst2
from util.log import init_logging
from util.model import load_roberta

init_logging()

logging.info(f"PyTorch version: {torch.__version__}")
logging.info(f"Numpy version: {np.__version__}")

sst2_dataset = load_sst2()

roberta = load_roberta()
