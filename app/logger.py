"""Application-wide logging configuration."""
import logging
import sys


def configure_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if root.handlers:
        # Already configured (e.g. re-imported under uvicorn --reload)
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.setLevel(level)
    root.addHandler(handler)
    # Quiet down noisy third-party loggers
    for noisy in ("httpx", "chromadb", "sentence_transformers", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
