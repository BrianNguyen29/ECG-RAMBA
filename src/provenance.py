"""Stable provenance fingerprints shared by training and evaluation."""

from __future__ import annotations

import hashlib

import numpy as np


def record_order_fingerprint(record_ids: np.ndarray) -> str:
    """Hash record identifiers in order with unambiguous length framing."""
    digest = hashlib.sha256()
    for value in np.asarray(record_ids).astype(str):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, byteorder="little", signed=False))
        digest.update(encoded)
    return digest.hexdigest()[:16]
