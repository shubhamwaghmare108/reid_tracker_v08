"""Centralized Re-ID embedding validation for V08.

This module is intentionally small and dependency-free apart from NumPy so
callers can validate descriptors before they are allowed to mutate track or
gallery state.
"""
from __future__ import annotations
import logging
import numpy as np

logger = logging.getLogger(__name__)


def validate_embedding(embedding, expected_dim: int | None = None):
    """Return ``(valid, embedding, reason)`` without ever returning bad data."""
    if embedding is None:
        return False, None, "NONE"
    try:
        value = np.asarray(embedding)
    except Exception:
        return False, None, "MALFORMED"
    if value.size == 0:
        return False, None, "EMPTY"
    if not np.issubdtype(value.dtype, np.floating):
        return False, None, "UNEXPECTED_DTYPE"
    value = value.reshape(-1)
    if expected_dim is not None and value.shape[0] != expected_dim:
        return False, None, "WRONG_DIMENSION"
    if not np.isfinite(value).all():
        return False, None, "NAN_OR_INF"
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm):
        return False, None, "NAN_OR_INF"
    if norm <= 1e-8:
        return False, None, "ZERO_NORM"
    return True, value.astype(np.float32, copy=False), "OK"


def safe_embedding(embedding, previous=None, expected_dim=None):
    """Return a valid descriptor, otherwise preserve the previous valid one."""
    if expected_dim is None and previous is not None:
        ok, prev, _ = validate_embedding(previous)
        if ok:
            expected_dim = prev.shape[0]
    ok, value, reason = validate_embedding(embedding, expected_dim)
    if ok:
        return value, True, reason
    logger.debug("Rejected ReID embedding: %s", reason)
    return previous, False, reason
