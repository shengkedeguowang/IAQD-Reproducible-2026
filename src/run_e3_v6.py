from __future__ import annotations

import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace
from qiskit_aer import AerSimulator

from common_v6 import PROCESSED, RAW, runtime_config, seeds, sha256_file, summary_row, write_csv


KET0 = np.array([1.0, 0.0], dtype=complex)
KET1 = np.array([0.0, 1.0], dtype=complex)
BB84 = {
    "0": KET0,
    "1": KET1,
    "+": (KET0 + KET1) / math.sqrt(2),
    "-": (KET0 - KET1) / math.sqrt(2),
}


def probe_unitary(theta: float) -> np.ndarray:
    rotation = np.array(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
        dtype=complex,
    )
    p0 = np.outer(KET0, KET0.conj())
    p1 = np.outer(KET1, KET1.conj())
    return np.kron(p0, np.eye(2)) + np.kron(p1, rotation)


def expected_joint(theta: float, label: str) -> np.ndarray:
    eve0 = KET0
    eve_theta = math.cos(theta) * KET0 + math.sin(theta) * KET1
    if label == "0":
        return np.kron(KET0, eve0)
    if label == "1":
        return np.kron(KET1, eve_theta)
    sign = 1 if label == "+" else -1
    return (np.kron(KET0, eve0) + sign * np.kron(KET1, eve_theta)) / math.sqrt(2)


def partial_trace_system(joint_density: np.ndarray) -> np.ndarray:
    tensor = joint_density.reshape(2, 2, 2, 2)
    return np.trace(tensor, axis1=1, axis2=3)


def partial_trace_eve(joint_density: np.ndarray) -> np.ndarray:
    tensor = joint_density.reshape(2, 2, 2, 2)
    return np.trace(tensor, axis1=0, axis2=2)


def trace_distance(left: np.ndarray, right: np.ndarray) -> float:
    singular_values = np.linalg.svd(left - right, compute_uv=False)
    return 0.5 * float(np.sum(singular_values))


def density_metrics(theta: float) -> dict[str, float]:
    unitary = probe_unitary(theta)
    identity_error = float(np.max(np.abs(unitary.conj().T @ unitary - np.eye(4))))
    qber_terms: list[float] = []
    eve_states: dict[str, np.ndarray] = {}
    evolution_error = 0.0
    for label, system_state in BB84.items():
        initial = np.kron(system_state, KET0)
        evolved = unitary @ initial
        expected = expected_joint(theta, label)
        phase = np.vdot(expected, evolved)
        if abs(phase) > 0:
            evolved = evolved * np.exp(-1j * np.angle(phase))
        evolution_error = max(evolution_error, float(np.max(np.abs(evolved - expected))))
        density = np.outer(evolved, evolved.conj())
        system_density = partial_trace_system(density)
        eve_states[label] = partial_trace_eve(density)
        correct_probability = float(np.real(np.vdot(system_state, system_density @ system_state)))
        qber_terms.append(1 - correct_probability)
    distance = trace_distance(eve_states["0"], eve_states["1"])
    return {
        "unitarity_error": identity_error,
        "bb84_evolution_error": evolution_error,
        "qber_density": float(np.mean(qber_terms)),
        "trace_distance_nuclear_norm": distance,
        "helstrom_guess": 0.5 * (1 + distance),
    }


def qiskit_metrics(theta: float) -> dict[str, float]:
    qber_terms: list[float] = []
    eve_states = {}
    for label in BB84:
        circuit = QuantumCircuit(2)
        if label == "1":
            circuit.x(0)
        elif label == "+":
            circuit.h(0)
        elif label == "-":
            circuit.x(0)
            circuit.h(0)
        circuit.cry(2 * theta, 0, 1)
        state = Statevector.from_instruction(circuit)
        system = np.asarray(partial_trace(state, [1]).data)
        eve_states[label] = np.asarray(partial_trace(state, [0]).data)
        reference = BB84[label]
        qber_terms.append(1 - float(np.real(np.vdot(reference, system @ reference))))
    distance = trace_distance(eve_states["0"], eve_states["1"])
    return {"qber_qiskit": float(np.mean(qber_terms)), "trace_distance_qiskit": distance, "helstrom_qiskit": 0.5 * (1 + distance)}


def qiskit_shot_qber(theta: float, shots: int) -> tuple[int, int]:
    circuits: list[QuantumCircuit] = []
    expected: list[str] = []
    for label in BB84:
        circuit = QuantumCircuit(2, 1)
        if label == "1":
            circuit.x(0)
        elif label == "+":
            circuit.h(0)
        elif label == "-":
            circuit.x(0)
            circuit.h(0)
        circuit.cry(2 * theta, 0, 1)
        if label in {"+", "-"}:
            circuit.h(0)
        circuit.measure(0, 0)
        circuits.append(circuit)
        expected.append("0" if label in {"0", "+"} else "1")
    result = AerSimulator().run(circuits, shots=shots, seed_simulator=seeds()[0]).result()
    errors = sum(shots - int(result.get_counts(index).get(wanted, 0)) for index, wanted in enumerate(expected))
    return errors, shots * len(circuits)


def run() -> list[dict[str, object]]:
    config = runtime_config()["e3"]
    grid = list(np.linspace(0.0, math.pi / 2, int(config["theta_grid_points"])))
    rng = np.random.default_rng(seeds()[0])
    random_angles = list(rng.uniform(0.0, math.pi / 2, int(config["random_crosscheck_angles"])))
    rows: list[dict[str, object]] = []
    for source, angles in (("grid", grid), ("random_non_grid", random_angles)):
        for theta in angles:
            density = density_metrics(float(theta))
            qiskit = qiskit_metrics(float(theta))
            qber_ref = (1 - math.cos(float(theta))) / 4
            distance_ref = math.sin(float(theta))
            guess_ref = (1 + distance_ref) / 2
            rows.append(
                {
                    "angle_source": source,
                    "theta": theta,
                    **density,
                    **qiskit,
                    "qber_analytic": qber_ref,
                    "trace_distance_analytic": distance_ref,
                    "helstrom_analytic": guess_ref,
                    "maximum_component_error": max(
                        density["unitarity_error"],
                        density["bb84_evolution_error"],
                        abs(density["qber_density"] - qber_ref),
                        abs(density["trace_distance_nuclear_norm"] - distance_ref),
                        abs(density["helstrom_guess"] - guess_ref),
                        abs(qiskit["qber_qiskit"] - qber_ref),
                        abs(qiskit["trace_distance_qiskit"] - distance_ref),
                    ),
                    "scope": "specified_U_theta_probe_family_only",
                }
            )
    if max(float(row["unitarity_error"]) for row in rows) > 1e-12:
        raise AssertionError("U_theta failed unitarity")
    if max(float(row["bb84_evolution_error"]) for row in rows) > 1e-12:
        raise AssertionError("BB84 evolution does not match the paper-defined probe")

    shot_rows = []
    for theta in (0.0, math.pi / 8, math.pi / 4, 3 * math.pi / 8, math.pi / 2):
        errors, total = qiskit_shot_qber(theta, int(config["qiskit_shots"]))
        shot_rows.append(
            {
                "theta": theta,
                "shots_per_bb84_state": int(config["qiskit_shots"]),
                "total_shots": total,
                "errors": errors,
                "qber_shots": errors / total,
                "qber_analytic": (1 - math.cos(theta)) / 4,
                "seed_simulator": seeds()[0],
            }
        )
    raw_path = RAW / "e3_probe_density_qiskit_crosscheck_v6.csv"
    shot_path = RAW / "e3_probe_qiskit_shots_v6.csv"
    write_csv(raw_path, rows)
    write_csv(shot_path, shot_rows)
    pi4 = min((row for row in rows if row["angle_source"] == "grid"), key=lambda row: abs(float(row["theta"]) - math.pi / 4))
    maximum_error = max(float(row["maximum_component_error"]) for row in rows)
    summaries = [
        summary_row("E3", "probe_theta_pi_over_4", "qber", pi4["qber_density"], unit="probability", raw_hash=sha256_file(raw_path), notes="Independent NumPy density-matrix path; specified U_theta family only."),
        summary_row("E3", "probe_theta_pi_over_4", "trace_distance", pi4["trace_distance_nuclear_norm"], unit="probability", raw_hash=sha256_file(raw_path)),
        summary_row("E3", "probe_theta_pi_over_4", "helstrom_guess_probability", pi4["helstrom_guess"], unit="probability", raw_hash=sha256_file(raw_path)),
        summary_row("E3", "grid_and_random_angles", "maximum_absolute_error", maximum_error, unit="probability", raw_hash=sha256_file(raw_path), notes="Includes unitarity, all four BB84 inputs, random non-grid angles, density matrices and independent Qiskit path."),
    ]
    write_csv(PROCESSED / "e3_summary_v6.csv", summaries)
    return summaries


if __name__ == "__main__":
    run()

