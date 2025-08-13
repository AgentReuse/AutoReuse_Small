## Package: transit_intent
# Directory structure:
# transit_intent/
# ├── __init__.py
# └── inference.py

# file: transit_intent/__init__.py
"""
transit_intent
=============
Simple package to perform intent classification and entity extraction
using pretrained BERT models.
"""
from .inference import load_models, predict

__all__ = ["load_models", "predict"]
