"""Core package initialization.

Import the ReID safety layer before tracker consumers construct tracks. The
layer validates descriptors and prevents invalid extraction results from
overwriting valid track embeddings.
"""
from app.core import reid_safety as _reid_safety
