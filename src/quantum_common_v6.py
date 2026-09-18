from __future__ import annotations

import math

import numpy as np


BELL_LABELS = ("00", "01", "10", "11")


def xor_label(left: str, right: str) -> str:
    if len(left) != 2 or len(right) != 2 or set(left + right) - {"0", "1"}:
        raise ValueError("Bell/Pauli labels must be two bits")
    return "".join(str(int(a) ^ int(b)) for a, b in zip(left, right))


def bell_vector(label: str) -> np.ndarray:
    """Generic Bell-label convention shared by independent implementations."""
    vectors = {
        "00": np.array([1, 0, 0, 1], complex) / math.sqrt(2),
        "01": np.array([0, 1, 1, 0], complex) / math.sqrt(2),
        "10": np.array([1, 0, 0, -1], complex) / math.sqrt(2),
        "11": np.array([0, -1, 1, 0], complex) / math.sqrt(2),
    }
    return vectors[label].copy()
