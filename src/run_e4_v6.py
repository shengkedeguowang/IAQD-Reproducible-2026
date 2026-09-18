from __future__ import annotations

import itertools
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Statevector, state_fidelity
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    ReadoutError,
    amplitude_damping_error,
    depolarizing_error,
    pauli_error,
    phase_damping_error,
)

from common_v6 import FIGURES, PROCESSED, RAW, runtime_config, seeds, sha256_file, summary_row, write_csv
from bell_truth_table_v6 import branch_fixture, expected_observations, recover_messages
from cluster_analytic_v6 import analytic_cluster_target, analytic_encoded_target
from cluster_circuit_v6 import assert_no_state_injection, explicit_cluster_circuit, full_protocol_circuit
from protocol_v6 import IAQDReferenceExecutor
from quantum_common_v6 import BELL_LABELS


STABILIZER_GENERATORS = {
    "phi1": ["-IIIIYY", "+YYIIII", "-IIIZXZ", "+IIXYIY", "+IYYZII", "+XZXIII"],
    "phi2": ["+IIIIYY", "+YYIIII", "+IIIZXZ", "-IIXYIY", "+IYYZII", "+XZXIII"],
}

def labels_from_count_key(key: str) -> tuple[str, str, str]:
    bits = key.replace(" ", "")[::-1]
    return bits[0:2], bits[2:4], bits[4:6]


def bell_decode_probabilities(density: DensityMatrix) -> np.ndarray:
    decode = QuantumCircuit(6)
    for first, second in ((0, 1), (2, 3), (4, 5)):
        decode.cx(first, second)
        decode.h(first)
    return np.asarray(density.evolve(decode).probabilities())


def probability_metrics(initial_state: str, alice: str, bob: str, probabilities: np.ndarray) -> tuple[float, float, float, float]:
    valid = alice_correct = bob_correct = both = 0.0
    for index, probability in enumerate(probabilities):
        if probability <= 0:
            continue
        observed = labels_from_count_key(format(index, "06b"))
        if observed in expected_observations(initial_state, alice, bob):
            valid += float(probability)
        recovered_alice, recovered_bob, _ = recover_messages(initial_state, observed, known_alice=alice, known_bob=bob)
        if recovered_alice == alice:
            alice_correct += float(probability)
        if recovered_bob == bob:
            bob_correct += float(probability)
        if recovered_alice == alice and recovered_bob == bob:
            both += float(probability)
    return valid, alice_correct, bob_correct, both


def noise_channel(name: str, probability: float):
    if name == "bit_flip":
        return pauli_error([("X", probability), ("I", 1 - probability)]).to_quantumchannel()
    if name == "phase_flip":
        return pauli_error([("Z", probability), ("I", 1 - probability)]).to_quantumchannel()
    if name == "depolarizing":
        return depolarizing_error(probability, 1).to_quantumchannel()
    if name == "amplitude_damping":
        return amplitude_damping_error(probability).to_quantumchannel()
    if name == "phase_damping":
        return phase_damping_error(probability).to_quantumchannel()
    raise ValueError(name)


def apply_noise_to_qubits(state: Statevector, channel_name: str, probability: float, qubits: tuple[int, ...]) -> DensityMatrix:
    density = DensityMatrix(state)
    channel = noise_channel(channel_name, probability)
    for qubit in qubits:
        density = density.evolve(channel, qargs=[qubit])
    return density


def noise_model(single_qubit: float, two_qubit: float, readout: float) -> NoiseModel:
    model = NoiseModel()
    model.add_all_qubit_quantum_error(depolarizing_error(single_qubit, 1), ["h", "x", "y", "z", "s", "sdg"])
    model.add_all_qubit_quantum_error(depolarizing_error(two_qubit, 2), ["cx", "cz"])
    model.add_all_qubit_readout_error(ReadoutError([[1 - readout, readout], [readout, 1 - readout]]))
    return model


def shot_metrics(counts: dict[str, int], shots: int, initial_state: str, alice: str, bob: str) -> tuple[float, float, float, float]:
    valid = alice_correct = bob_correct = both = 0
    for key, count in counts.items():
        observed = labels_from_count_key(key)
        if observed in expected_observations(initial_state, alice, bob):
            valid += count
        recovered_alice, recovered_bob, _ = recover_messages(initial_state, observed, known_alice=alice, known_bob=bob)
        alice_correct += count if recovered_alice == alice else 0
        bob_correct += count if recovered_bob == bob else 0
        both += count if recovered_alice == alice and recovered_bob == bob else 0
    return valid / shots, alice_correct / shots, bob_correct / shots, both / shots


def draw_circuits() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    for name in ("phi1", "phi2"):
        circuit = explicit_cluster_circuit(name)
        figure, axis = plt.subplots(figsize=(18, 4.4))
        axis.set_title(f"Explicit Clifford preparation of {name}: H/S/X/Y/CX only", fontsize=12)
        for qubit in range(6):
            y = 5 - qubit
            axis.plot([0, len(circuit.data) + 1], [y, y], color="black", linewidth=0.8)
            axis.text(-0.5, y, f"q{qubit}", ha="right", va="center", fontsize=9)
        for column, item in enumerate(circuit.data, start=1):
            operation = item.operation.name
            qubits = [circuit.find_bit(qubit).index for qubit in item.qubits]
            if operation == "cx":
                control_y = 5 - qubits[0]
                target_y = 5 - qubits[1]
                axis.plot([column, column], [control_y, target_y], color="#1f4e79", linewidth=1.0)
                axis.add_patch(Circle((column, control_y), 0.08, color="#1f4e79"))
                axis.add_patch(Circle((column, target_y), 0.18, fill=False, edgecolor="#1f4e79", linewidth=1.1))
                axis.plot([column - 0.13, column + 0.13], [target_y, target_y], color="#1f4e79", linewidth=1.0)
                axis.plot([column, column], [target_y - 0.13, target_y + 0.13], color="#1f4e79", linewidth=1.0)
            else:
                y = 5 - qubits[0]
                axis.add_patch(Rectangle((column - 0.25, y - 0.22), 0.5, 0.44, facecolor="#e8f1f8", edgecolor="#1f4e79", linewidth=0.8))
                axis.text(column, y, operation.upper(), ha="center", va="center", fontsize=7)
        axis.set_xlim(-1.0, len(circuit.data) + 1.0)
        axis.set_ylim(-0.7, 5.7)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_frame_on(False)
        figure.tight_layout()
        figure.savefig(FIGURES / f"e4_explicit_{name}_circuit_v6.png", dpi=300, bbox_inches="tight")
        plt.close(figure)


def run() -> list[dict[str, object]]:
    config = runtime_config()["e4"]
    shots = int(config["qiskit_shots"])
    draw_circuits()
    circuit_rows: list[dict[str, object]] = []
    exact_rows: list[dict[str, object]] = []
    actual_supports: dict[tuple[str, str, str], list[tuple[str, str, str]]] = {}
    measured_circuits: list[QuantumCircuit] = []
    metadata: list[tuple[str, str, str]] = []
    for initial_state in ("phi1", "phi2"):
        preparation = explicit_cluster_circuit(initial_state)
        assert_no_state_injection(preparation)
        operations = preparation.count_ops()
        forbidden = {name for name in operations if name.lower() in {"initialize", "state_preparation", "unitary", "isometry"}}
        if forbidden:
            raise AssertionError(f"forbidden preparation operations: {forbidden}")
        actual_preparation = Statevector.from_instruction(preparation)
        target_preparation = Statevector(analytic_cluster_target(initial_state))
        prep_fidelity = float(state_fidelity(actual_preparation, target_preparation))
        if prep_fidelity < 1 - 1e-12:
            raise AssertionError(f"explicit {initial_state} circuit disagrees with Bell expansion")
        nonzero_actual = np.flatnonzero(np.abs(actual_preparation.data) > 1e-12)
        nonzero_target = np.flatnonzero(np.abs(target_preparation.data) > 1e-12)
        phase = np.vdot(target_preparation.data, actual_preparation.data)
        aligned = actual_preparation.data * np.exp(-1j * np.angle(phase))
        amplitude_error = float(np.max(np.abs(aligned - target_preparation.data)))
        circuit_rows.append(
            {
                "initial_state": initial_state,
                "stabilizer_generators": " ".join(STABILIZER_GENERATORS[initial_state]),
                "gate_sequence": ";".join(item.operation.name + "(" + ",".join(str(preparation.find_bit(q).index) for q in item.qubits) + ")" for item in preparation.data),
                "single_qubit_gates": sum(count for name, count in operations.items() if name not in {"cx", "cz"}),
                "two_qubit_gates": int(operations.get("cx", 0)) + int(operations.get("cz", 0)),
                "depth": preparation.depth(),
                "measurements": 0,
                "nonzero_amplitudes_actual": len(nonzero_actual),
                "nonzero_amplitudes_target": len(nonzero_target),
                "support_matches": set(nonzero_actual) == set(nonzero_target),
                "maximum_aligned_amplitude_error": amplitude_error,
                "preparation_fidelity": prep_fidelity,
            }
        )
        for alice, bob in itertools.product(BELL_LABELS, BELL_LABELS):
            circuit = full_protocol_circuit(initial_state, alice, bob, measure=False)
            assert_no_state_injection(circuit)
            actual = Statevector.from_instruction(circuit)
            target = Statevector(analytic_encoded_target(initial_state, alice, bob))
            fidelity = float(state_fidelity(actual, target))
            probabilities = bell_decode_probabilities(DensityMatrix(actual))
            actual_support = sorted(
                labels_from_count_key(format(index, "06b"))
                for index, probability in enumerate(probabilities)
                if float(probability) > 1e-12
            )
            if set(actual_support) != expected_observations(initial_state, alice, bob):
                raise AssertionError("Qiskit statevector Bell support disagrees with independent truth table")
            actual_supports[(initial_state, alice, bob)] = actual_support
            valid, alice_correct, bob_correct, both = probability_metrics(initial_state, alice, bob, probabilities)
            exact_rows.append(
                {
                    "initial_state": initial_state,
                    "alice_message": alice,
                    "bob_message": bob,
                    "state_fidelity": fidelity,
                    "bell_label_probability": valid,
                    "alice_exact_decode_probability": alice_correct,
                    "bob_exact_decode_probability": bob_correct,
                    "bidirectional_exact_decode_probability": both,
                    "num_qubits": 6,
                    "joint_6n_state_constructed": False,
                }
            )
            measured_circuits.append(full_protocol_circuit(initial_state, alice, bob, measure=True))
            metadata.append((initial_state, alice, bob))
    if min(float(row["state_fidelity"]) for row in exact_rows) < 1 - 1e-12:
        raise AssertionError("an encoded explicit circuit disagrees with analytic target")
    if min(float(row["bidirectional_exact_decode_probability"]) for row in exact_rows) < 1 - 1e-12:
        raise AssertionError("not all 32 exact circuits recover both messages")

    branch_rows: list[dict[str, object]] = []
    branch_index = 0
    for initial_state, alice, bob in metadata:
        for observed in actual_supports[(initial_state, alice, bob)]:
            recovered_alice, recovered_bob, intermediates = recover_messages(
                initial_state, observed, known_alice=alice, known_bob=bob
            )
            quantum_block = branch_fixture(initial_state, alice, bob, observed)
            quantum_block["source"] = "qiskit_statevector_nonzero_bell_support"
            execution = IAQDReferenceExecutor(seeds()[0] + branch_index).run_basic(
                alice, bob, quantum_blocks=[quantum_block]
            )
            algebra_event = next(item for item in execution.trace if item["event"] == "consume_external_bell_decode")
            row = {
                "branch_id": f"{initial_state}-A{alice}-B{bob}-{'_'.join(observed)}",
                "branch_index": branch_index,
                "initial_state": initial_state,
                "alice_message": alice,
                "bob_message": bob,
                "bell_branch_12": observed[0],
                "bell_branch_34": observed[1],
                "bell_branch_56": observed[2],
                "M_B": intermediates["M_B"],
                "tilde_M_B": intermediates["tilde_M_B"],
                "M_A": intermediates["M_A"],
                "tilde_M_A": intermediates["tilde_M_A"],
                "K_A_otp_hex": algebra_event["K_A_otp"],
                "K_B_otp_hex": algebra_event["K_B_otp"],
                "C_A_hex": algebra_event["C_A"],
                "C_B_hex": algebra_event["C_B"],
                "ciphertext_A_formula": "C_A=tilde_M_A XOR K_A_otp",
                "ciphertext_B_formula": "C_B=tilde_M_B XOR K_B_otp",
                "ciphertext_formulas_valid": (
                    int(algebra_event["C_A"], 16) == (int(intermediates["tilde_M_A"], 2) ^ int(algebra_event["K_A_otp"], 16))
                    and int(algebra_event["C_B"], 16) == (int(intermediates["tilde_M_B"], 2) ^ int(algebra_event["K_B_otp"], 16))
                ),
                "message_relations_valid": (
                    int(intermediates["M_A"], 2) ^ int(intermediates["tilde_M_A"], 2) == int(alice, 2)
                    and int(intermediates["M_B"], 2) ^ int(intermediates["tilde_M_B"], 2) == int(bob, 2)
                ),
                "decoded_intermediates": __import__("json").dumps(intermediates, sort_keys=True),
                "passed_classical_quantum_input": __import__("json").dumps(execution.quantum_inputs[0], sort_keys=True),
                "Out_A": recovered_bob,
                "Out_B": recovered_alice,
                "executor_recovered_by_alice": None if execution.recovered_by_alice is None else f"{int.from_bytes(execution.recovered_by_alice, 'big'):02b}"[-2:],
                "executor_recovered_by_bob": None if execution.recovered_by_bob is None else f"{int.from_bytes(execution.recovered_by_bob, 'big'):02b}"[-2:],
                "gamma_A_valid": execution.gamma_a_valid,
                "gamma_B_valid": execution.gamma_b_valid,
                "terminal_state": execution.terminal_state,
                "logical_erasure": execution.logical_erasure,
                "branch_success": (
                    recovered_alice == alice
                    and recovered_bob == bob
                    and execution.success
                    and execution.terminal_state == "COMPLETED"
                    and execution.logical_erasure
                    and execution.quantum_inputs[0]["bell_labels"] == list(observed)
                    and algebra_event["M_A"] == intermediates["M_A"]
                    and algebra_event["M_B"] == intermediates["M_B"]
                    and algebra_event["tilde_M_A"] == intermediates["tilde_M_A"]
                    and algebra_event["tilde_M_B"] == intermediates["tilde_M_B"]
                ),
            }
            branch_rows.append(row)
            branch_index += 1
    if len(branch_rows) != 128 or not all(bool(row["branch_success"]) for row in branch_rows):
        raise AssertionError("the 128 exact Bell branches did not all complete the IAQD reference execution")

    ideal_result = AerSimulator().run(measured_circuits, shots=shots, seed_simulator=seeds()[0]).result()
    shot_rows: list[dict[str, object]] = []
    for index, (initial_state, alice, bob) in enumerate(metadata):
        valid, alice_rate, bob_rate, both = shot_metrics(ideal_result.get_counts(index), shots, initial_state, alice, bob)
        shot_rows.append(
            {
                "initial_state": initial_state,
                "alice_message": alice,
                "bob_message": bob,
                "shots": shots,
                "bell_label_recovery_rate": valid,
                "alice_decode_rate": alice_rate,
                "bob_decode_rate": bob_rate,
                "bidirectional_decode_rate": both,
                "acceptance_rate": both,
                "honest_false_abort": 1 - both,
                "seed_simulator": seeds()[0],
            }
        )

    channel_names = ("bit_flip", "phase_flip", "depolarizing", "amplitude_damping", "phase_damping")
    channel_rows: list[dict[str, object]] = []
    for (initial_state, alice, bob), exact in zip(metadata, exact_rows):
        ideal = Statevector(analytic_encoded_target(initial_state, alice, bob))
        for noise_scope, qubits in (("transmission_S3_S4_only", (2, 3)), ("local_storage_S1_S2_S5_S6_only", (0, 1, 4, 5))):
            for channel_name in channel_names:
                for probability in config["channel_p_values"]:
                    density = apply_noise_to_qubits(ideal, channel_name, float(probability), qubits)
                    valid, alice_rate, bob_rate, both = probability_metrics(
                        initial_state, alice, bob, bell_decode_probabilities(density)
                    )
                    channel_rows.append(
                        {
                            "initial_state": initial_state,
                            "alice_message": alice,
                            "bob_message": bob,
                            "noise_class": "A_transmission_channel" if noise_scope.startswith("transmission") else "B_local_storage",
                            "noise_scope": noise_scope,
                            "affected_qubits": ",".join(str(q) for q in qubits),
                            "channel": channel_name,
                            "p": probability,
                            "state_fidelity_to_ideal": float(state_fidelity(density, ideal)),
                            "bell_label_probability": valid,
                            "alice_decode_rate": alice_rate,
                            "bob_decode_rate": bob_rate,
                            "acceptance_rate": both,
                            "honest_false_abort": 1 - both,
                        }
                    )

    gate_rows: list[dict[str, object]] = []
    for profile, values in config["gate_noise_profiles"].items():
        simulator = AerSimulator(noise_model=noise_model(float(values["single_qubit"]), float(values["two_qubit"]), float(values["readout"])))
        result = simulator.run(measured_circuits, shots=shots, seed_simulator=seeds()[0] + len(gate_rows)).result()
        for index, (initial_state, alice, bob) in enumerate(metadata):
            valid, alice_rate, bob_rate, both = shot_metrics(result.get_counts(index), shots, initial_state, alice, bob)
            gate_rows.append(
                {
                    "profile": profile,
                    "initial_state": initial_state,
                    "alice_message": alice,
                    "bob_message": bob,
                    "single_qubit_error": values["single_qubit"],
                    "two_qubit_error": values["two_qubit"],
                    "readout_error": values["readout"],
                    "noise_class": "C_gate_and_measurement",
                    "affected_operations": "h,x,y,z,s,sdg,cx,cz,measure",
                    "shots": shots,
                    "bell_label_recovery_rate": valid,
                    "alice_decode_rate": alice_rate,
                    "bob_decode_rate": bob_rate,
                    "acceptance_rate": both,
                    "honest_false_abort": 1 - both,
                }
            )

    circuit_path = RAW / "e4_explicit_circuit_inventory_v6.csv"
    exact_path = RAW / "e4_exact_32_messages_v6.csv"
    shot_path = RAW / "e4_8192_shots_32_messages_v6.csv"
    channel_path = RAW / "e4_transmission_storage_noise_v6.csv"
    gate_path = RAW / "e4_gate_measurement_noise_v6.csv"
    branch_path = RAW / "e4_basic_end_to_end_128_branches_v6.csv"
    write_csv(circuit_path, circuit_rows)
    write_csv(exact_path, exact_rows)
    write_csv(shot_path, shot_rows)
    write_csv(channel_path, channel_rows)
    write_csv(gate_path, gate_rows)
    write_csv(branch_path, branch_rows)

    def mean_channel(scope: str, channel: str, probability: float, field: str) -> float:
        values = [float(row[field]) for row in channel_rows if row["noise_class"] == scope and row["channel"] == channel and math.isclose(float(row["p"]), probability)]
        return float(np.mean(values))

    medium_gate = [float(row["acceptance_rate"]) for row in gate_rows if row["profile"] == "medium"]
    summaries = [
        summary_row("E4", "phi1_explicit_preparation", "state_fidelity", circuit_rows[0]["preparation_fidelity"], unit="probability", raw_hash=sha256_file(circuit_path), notes="Hard-coded H/S/X/Y/CX stabilizer-synthesis circuit versus independent Bell expansion."),
        summary_row("E4", "phi2_explicit_preparation", "state_fidelity", circuit_rows[1]["preparation_fidelity"], unit="probability", raw_hash=sha256_file(circuit_path)),
        summary_row("E4", "all_32_explicit_circuits", "bidirectional_exact_decode_accuracy", min(float(row["bidirectional_exact_decode_probability"]) for row in exact_rows), unit="probability", raw_hash=sha256_file(exact_path)),
        summary_row("E4", "basic_mode_128_exact_branches", "successful_exact_branches", sum(bool(row["branch_success"]) for row in branch_rows), unit="branches", raw_hash=sha256_file(branch_path), notes="128=2 cluster states x 16 message pairs x 4 nonzero Bell branches; each branch reaches COMPLETED with both gamma checks and logical erasure."),
        summary_row("E4", "all_32_8192_shots", "bidirectional_shot_decode_accuracy", float(np.mean([float(row["bidirectional_decode_rate"]) for row in shot_rows])), unit="probability", raw_hash=sha256_file(shot_path)),
        summary_row("E4", "transmission_depolarizing_p010", "acceptance_rate", mean_channel("A_transmission_channel", "depolarizing", 0.10, "acceptance_rate"), unit="probability", raw_hash=sha256_file(channel_path), notes="Noise applied only to transmitted S3,S4 qubits 2,3."),
        summary_row("E4", "storage_amplitude_damping_p010", "honest_false_abort", mean_channel("B_local_storage", "amplitude_damping", 0.10, "honest_false_abort"), unit="probability", raw_hash=sha256_file(channel_path), notes="Storage noise applied only to local S1,S2,S5,S6 qubits 0,1,4,5."),
        summary_row("E4", "gate_measurement_medium", "acceptance_rate", float(np.mean(medium_gate)), unit="probability", raw_hash=sha256_file(gate_path), notes="Aer NoiseModel on actual gates plus separate readout error."),
    ]
    write_csv(PROCESSED / "e4_summary_v6.csv", summaries)
    return summaries


if __name__ == "__main__":
    run()

