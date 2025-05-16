import logging
import os
import sys
from pathlib import Path
from typing import Literal, Optional


class ColorFormatter(logging.Formatter):
    """Custom formatter with colors"""

    COLORS = {
        "DEBUG": "\033[37m",  # White
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[41m",  # Red background
    }
    RESET = "\033[0m"

    def format(self, record):
        # Add color to the level name
        color = self.COLORS.get(record.levelname, "")
        record.levelname = f"{color}{record.levelname}{self.RESET}"

        return super().format(record)


def get_logger(
    name: str = None,
    level: Optional[
        int | Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    ] = None,
) -> logging.Logger:
    """
    Create a logger with consistent formatting including filename and line number

    Args:
        name: Logger name, defaults to file name if None

    Returns:
        Configured logger instance
    """
    if name is None:
        # Get the caller's filename if no name provided
        frame = sys._getframe(1)
        name = Path(frame.f_code.co_filename).stem

    # Create logger
    logger = logging.getLogger(name)

    # Get the level from the environment variables LOG_LEVEL or
    if level is None:
        if "LOG_LEVEL" in os.environ:
            level = os.environ["LOG_LEVEL"]
        elif "LOGLEVEL" in os.environ:
            level = os.environ["LOGLEVEL"]
        else:
            level = "INFO"

    if isinstance(level, str):
        level = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }[level]

    # Only add handler if logger doesn't have one
    if not logger.handlers:
        # Create stderr handler
        handler = logging.StreamHandler(sys.stderr)

        # Get CUDA_VISIBLE_DEVICES for logging
        cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        cuda_info = f"[CUDA:{cuda_devices}] " if cuda_devices else ""

        # Format: [LEVEL] [CUDA:x,y] filename:line - message
        formatter = ColorFormatter(
            fmt=f"[%(levelname)s] {cuda_info}%(filename)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Set default level to DEBUG to see all messages
        logger.setLevel(level)

        # Prevent propagation to avoid duplicate logs
        logger.propagate = False

    return logger
