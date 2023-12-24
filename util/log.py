import logging
import os

MAIN_LOG = "log/main.log"


def init_logging(title: str):
    level = os.environ.get("LOG_LEVEL")
    if level is None:
        level = logging.INFO
    os.makedirs("tmp", exist_ok=True)

    append_mode = os.path.isfile(MAIN_LOG)

    logging.basicConfig(
        level=level,
        filename=MAIN_LOG,
        filemode="a",
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if append_mode:
        logging.info("")
        logging.info("")
    logging.info(f"<Start {title}>")


def print_info(msg: object):
    print(f"[INFO] {msg}")
    logging.info(msg)
