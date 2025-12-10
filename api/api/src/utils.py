import logging
import torch
from nltk import word_tokenize

def check_num_tokens(
    document : str,
    min_tokens : int, 
    max_token : int,
) -> str:

    num_tokens = len(word_tokenize(document))

    if num_tokens < min_tokens:
        raise ValueError(f"Document contains {num_tokens} tokens which is lower than the lower bound {min_tokens}.")
        
    if num_tokens > max_token:
        raise ValueError(f"Document contains {num_tokens} tokens which is higher than the upper bound {min_tokens}.")
        
    return document

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


    return logger

def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

__loggers = {}

def get_logger(name : str) -> logging.Logger:

    if name not in __loggers:
        __loggers[name] = setup_logger(name)
    
    return __loggers[name]