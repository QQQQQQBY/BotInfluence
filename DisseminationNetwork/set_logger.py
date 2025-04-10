import os
import logging

def set_logger(log_file, name="default"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # output_folder = "output"
    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder)

    # log_folder = os.path.join(output_folder, "log")
    # if not os.path.exists(log_folder):
    #     os.makedirs(log_folder)

    # log_file = os.path.join(log_folder, log_file)

    handler = logging.FileHandler(log_file, mode="a", encoding='utf-8')
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(handler)
    return logger