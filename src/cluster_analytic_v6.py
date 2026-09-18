from __future__ import annotations

import math

import numpy as np

from quantum_common_v6 import bell_vector


# Paper Bell expansions, transcribed independently from the circuit synthesis.
_PHI1_EXPANSION = (
    ("00", "00", "01", +1),
    ("00", "01", "10", +1),
    ("11", "10", "01", -1),
    ("11", "11", "10", -1),
)
_PHI2_EXPANSION = (
    ("01", "00", "01", +1),
    ("01", "01", "10", +1),
    ("10", "10", "01", +1),
    ("10", "11", "10", +1),
)


def analytic_cluster_target(cluster_type: str) -> np.ndarray:
    terms = {"phi1": _PHI1_EXPANSION, "phi2": _PHI2_EXPANSION}[cluster_type]
    vector = np.zeros(64, dtype=complex)
    for label12, label34, label56, phase in terms:
        vector += phase * np.kron(bell_vector(label56), np.kron(bell_vector(label34), bell_vector(label12))) / 2
    if not math.isclose(float(np.linalg.norm(vector)), 1.0, abs_tol=1e-12):
        raise AssertionError("paper Bell expansion is not normalized")
    return vector


def _apply_one_qubit(vector: np.ndarray, operator: np.ndarray, qubit: int) -> np.ndarray:
    tensor = vector.reshape([2] * 6)
    axis = 5 - qubit
    moved = np.moveaxis(tensor, axis, 0)
    transformed = np.tensordot(operator, moved, axes=([1], [0]))
    return np.moveaxis(transformed, 0, axis).reshape(64)


def analytic_encoded_target(cluster_type: str, alice_message: str, bob_message: str) -> np.ndarray:
    vector = analytic_cluster_target(cluster_type)
    x = np.array([[0, 1], [1, 0]], complex)
    z = np.array([[1, 0], [0, -1]], complex)
    for qubit, label in ((2, bob_message), (4, alice_message)):
        if label[1] == "1":
            vector = _apply_one_qubit(vector, x, qubit)
        if label[0] == "1":
            vector = _apply_one_qubit(vector, z, qubit)
    return vector
