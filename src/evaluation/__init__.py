"""
Evaluation Module for Qwen3-0.6B Model
Refactored modular design
"""

from .config import *
from .data_loader import DataLoader
from .model_client import ModelClient
from .metrics_calculator import MetricsCalculator
from .file_handler import FileHandler
from .evaluate_qwen_model import EvaluationRunner, main

__version__ = "1.0.0"

__all__ = [
    # Configuration
    "DEFAULT_API_URL",
    "DEFAULT_API_KEY",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_INPUT_CSV_PATH",
    "DEFAULT_SAVE_PATH",
    "DEFAULT_METRICS_PATH",
    "DEFAULT_SYSTEM_PROMPT",
    "DEFAULT_MAX_WORKERS",
    "DEFAULT_REQUEST_TIMEOUT",

    # Core classes
    "DataLoader",
    "ModelClient",
    "MetricsCalculator",
    "FileHandler",
    "EvaluationRunner",

    # Main function
    "main"
]