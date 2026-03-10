"""Wildlife age and sex classification package."""

__version__ = "0.1.0"

from .io import load, load_model, save
from .preprocess import preprocess_data
from .train import split_data, tune_model, evaluate_model

__all__ = [
    "load",
    "load_model",
    "preprocess_data",
    "save",
    "split_data",
    "tune_model",
    "evaluate_model",
]
