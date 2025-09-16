import os
import logging


def dumps_matrix(mat: list[list[int]]) -> str:
    if not isinstance(mat, list) or not all(isinstance(row, list) for row in mat):
        return str(mat)
    return '[\n  ' + ',\n  '.join(str(row) for row in mat) + '\n]'


def setup_file_logger(name, log_file, level=logging.INFO):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)

    handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(handler)

    logger.propagate = False
    return logger
