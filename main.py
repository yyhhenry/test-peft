import torch
import numpy as np
import logging
from src.init_logging import init_logging

init_logging()

logging.info(f"PyTorch version: {torch.__version__}")
logging.info(f"Numpy version: {np.__version__}")
