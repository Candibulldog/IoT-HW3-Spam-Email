"""
Spam Detection Package

A machine learning-based spam email/SMS detection system.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .evaluate import evaluate_model
from .preprocess import TextPreprocessor, preprocess_text
from .utils import load_dataset, load_model, save_model

__all__ = ["TextPreprocessor", "preprocess_text", "evaluate_model", "load_model", "save_model", "load_dataset"]
