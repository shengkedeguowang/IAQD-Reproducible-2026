"""C quantum-core and reference-composed full noisy-protocol experiments."""

from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from qiskit.quantum_info import DensityMatrix, Statevector, state_fidelity
from qiskit_aer import AerSimulator

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

from bell_truth_table_v6 import branch_fixture, expected_observations, recover_messages  # noqa: E402
from cluster_analytic_v6 import analytic_cluster_target, analytic_encoded_target  # noqa: E402
from cluster_circuit_v6 import assert_no_state_injection, explicit_cluster_circuit, full_protocol_circuit  # noqa: E402
from protocol_v7 import BASIC, ID_A, ID_B, RELEASE, IAQDSession, ProtocolViolation, parse_record, peer_for  # noqa: E402
from quantum_common_v6 import BELL_LABELS  # noqa: E402
from release_v7_common import (  # noqa: E402
    DERIVED,
    RAW,
    SEEDS,
    clopper_pearson_upper_zero,
    deterministic_protocol_secrets,
    sha256_file,
    wilson_interval,
    write_csv,
    write_json,
)
from run_e4_v6 import (  # noqa: E402
    STABILIZER_GENERATORS,
    apply_noise_to_qubits,
    bell_decode_probabilities,
    labels_from_count_key,
    noise_model,
    probability_metrics,
    shot_metrics,
)


def finish_release(session: IAQDSession) -> None:
    commit_a = session.send_commit_set(ID_A)
    commit_b = session.send_commit_set(ID_B)
    if not session.deliver(commit_a, ID_B).accepted or not session.deliver(commit_b, ID_A).accepted:
        raise AssertionError("honest commit exchange failed")
    for j in range(1, int(session.params["r"]) + 1):
        leader = session.leader_for_block(j)
        follower = peer_for(leader)
        first = session.send_open(leader, j)
        if not session.deliver(first, follower).accepted:
            raise AssertionError("honest leader opening failed")
        second = session.send_open(follower, j)
        if not session.deliver(second, leader).accepted:
            raise AssertionError("honest follower opening failed")


def run_core() -> dict[str, Any]:
    circuit_rows: list[dict[str, Any]] = []
    exact_rows: list[dict[str, Any]] = []
    supports: dict[tuple[str, str, str], list[tuple[str, str, str]]] = {}
    measured = []
    metadata: list[tuple[str, str, str]] = []
    for cluster in ("phi1", "phi2"):
        preparation = explicit_cluster_circuit(cluster)
        assert_no_state_injection(preparation)
        actual = Statevector.from_instruction(preparation)
        target = Statevector(analytic_cluster_target(cluster))
        fidelity = float(state_fidelity(actual, target))
        phase = np.vdot(target.data, actual.data)
        aligned = actual.data * np.exp(-1j * np.angle(phase))
        ops = {str(key): int(value) for key, value in preparation.count_ops().items()}
        circuit_rows.append(
            {
                "cluster_state": cluster,
                "stabilizer_generators": " ".join(STABILIZER_GENERATORS[cluster]),
                "num_qubits": preparation.num_qubits,
                "single_qubit_gates": sum(value for key, value in ops.items() if key not in {"cx", "cz"}),
                "two_qubit_gates": int(ops.get("cx", 0)) + int(ops.get("cz", 0)),
                "depth": preparation.depth(),
                "preparation_fidelity": fidelity,
                "maximum_aligned_amplitude_error": float(np.max(np.abs(aligned - target.data))),
                "forbidden_state_injection": False,
            }
        )
        for message_a, message_b in itertools.product(BELL_LABELS, BELL_LABELS):
            circuit = full_protocol_circuit(cluster, message_a, message_b, measure=False)
            assert_no_state_injection(circuit)
            state = Statevector.from_instruction(circuit)
            encoded_target = Statevector(analytic_encoded_target(cluster, message_a, message_b))
            probabilities = bell_decode_probabilities(DensityMatrix(state))
            support = sorted(
                labels_from_count_key(format(index, "06b"))
                for index, probability in enumerate(probabilities)
                if float(probability) > 1e-12
            )
            supports[(cluster, message_a, message_b)] = support
            valid, alice_rate, bob_rate, both = probability_metrics(cluster, message_a, message_b, probabilities)
            exact_rows.append(
                {
                    "cluster_state": cluster,
                    "message_a": message_a,
                    "message_b": message_b,
                    "state_fidelity": float(state_fidelity(state, encoded_target)),
                    "valid_bell_support_probability": valid,
                    "alice_local_decode_probability": bob_rate,
                    "bob_local_decode_probability": alice_rate,
                    "bidirectional_decode_probability": both,
                    "nonzero_bell_branches": len(support),
                    "joint_6n_state_constructed": False,
                }
            )
            measured.append(full_protocol_circuit(cluster, message_a, message_b, measure=True))
            metadata.append((cluster, message_a, message_b))

    branch_rows: list[dict[str, Any]] = []
    branch_index = 0
    for cluster, message_a, message_b in metadata:
        for observed in supports[(cluster, message_a, message_b)]:
            fixture = branch_fixture(cluster, message_a, message_b, observed)
            with deterministic_protocol_secrets(SEEDS[0] + branch_index):
                execution = IAQDSession(
                    mode=BASIC,
                    message_a=message_a,
                    message_b=message_b,
                    quantum_blocks=[fixture],
                ).run_honest()
            recovered_a, recovered_b, intermediates = recover_messages(
                cluster, observed, known_alice=message_a, known_bob=message_b
            )
            branch_rows.append(
                {
                    "branch_index": branch_index,
                    "cluster_state": cluster,
                    "message_a": message_a,
                    "message_b": message_b,
                    "bell_12": observed[0],
                    "bell_34": observed[1],
                    "bell_56": observed[2],
                    "M_A": intermediates["M_A"],
                    "tilde_M_A": intermediates["tilde_M_A"],
                    "M_B": intermediates["M_B"],
                    "tilde_M_B": intermediates["tilde_M_B"],
                    "truth_table_recovered_alice_message": recovered_a,
                    "truth_table_recovered_bob_message": recovered_b,
                    "v7_recovered_by_alice": execution.recovered_by_alice,
                    "v7_recovered_by_bob": execution.recovered_by_bob,
                    "v7_success": execution.success,
                    "v7_terminal": execution.terminal_state,
                    "v7_prf_confirm_count": execution.metrics["prf_confirm_count"],
                    "branch_success": execution.success and execution.recovered_by_alice == message_b and execution.recovered_by_bob == message_a,
                }
            )
            branch_index += 1

    ideal_result = AerSimulator().run(measured, shots=8192, seed_simulator=SEEDS[0]).result()
    shot_rows: list[dict[str, Any]] = []
    for index, (cluster, message_a, message_b) in enumerate(metadata):
        valid, alice_rate, bob_rate, both = shot_metrics(ideal_result.get_counts(index), 8192, cluster, message_a, message_b)
        shot_rows.append(
            {
                "cluster_state": cluster,
                "message_a": message_a,
                "message_b": message_b,
                "shots": 8192,
                "seed": SEEDS[0],
                "valid_bell_support_rate": valid,
                "alice_local_decode_rate": bob_rate,
                "bob_local_decode_rate": alice_rate,
                "bidirectional_decode_rate": both,
                "sampling_error_note": "finite Aer shots; numerical exact rows are separate",
            }
        )

    channel_rows: list[dict[str, Any]] = []
    channels = ("bit_flip", "phase_flip", "depolarizing", "amplitude_damping", "phase_damping")
    p_values = (0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20)
    for cluster, message_a, message_b in metadata:
        ideal = Statevector(analytic_encoded_target(cluster, message_a, message_b))
        for scope, qubits in (("transmission_S3_S4_only", (2, 3)), ("local_storage_S1_S2_S5_S6_only", (0, 1, 4, 5))):
            for channel, p in itertools.product(channels, p_values):
                density = apply_noise_to_qubits(ideal, channel, p, qubits)
                valid, alice_rate, bob_rate, both = probability_metrics(
                    cluster, message_a, message_b, bell_decode_probabilities(density)
                )
                channel_rows.append(
                    {
                        "cluster_state": cluster,
                        "message_a": message_a,
                        "message_b": message_b,
                        "noise_scope": scope,
                        "affected_qubits": ",".join(map(str, qubits)),
                        "channel": channel,
                        "p": p,
                        "state_fidelity_to_ideal": float(state_fidelity(density, ideal)),
                        "valid_bell_support_probability": valid,
                        "alice_local_decode_probability": bob_rate,
                        "bob_local_decode_probability": alice_rate,
                        "bidirectional_decode_probability": both,
                        "local_decode_failure_probability": 1 - both,
                        "metric_scope": "single six-qubit block; not full-protocol acceptance",
                    }
                )

    profiles = {
        "low": {"single_qubit": 0.0005, "two_qubit": 0.005, "readout": 0.01},
        "medium": {"single_qubit": 0.001, "two_qubit": 0.01, "readout": 0.02},
        "high": {"single_qubit": 0.005, "two_qubit": 0.03, "readout": 0.05},
    }
    gate_rows: list[dict[str, Any]] = []
    for profile_index, (profile, values) in enumerate(profiles.items()):
        simulator = AerSimulator(noise_model=noise_model(values["single_qubit"], values["two_qubit"], values["readout"]))
        result = simulator.run(measured, shots=8192, seed_simulator=SEEDS[0] + profile_index).result()
        for index, (cluster, message_a, message_b) in enumerate(metadata):
            valid, alice_rate, bob_rate, both = shot_metrics(result.get_counts(index), 8192, cluster, message_a, message_b)
            low, high = wilson_interval(round(both * 8192), 8192)
            gate_rows.append(
                {
                    "profile": profile,
                    "cluster_state": cluster,
                    "message_a": message_a,
                    "message_b": message_b,
                    **values,
                    "shots": 8192,
                    "seed": SEEDS[0] + profile_index,
                    "valid_bell_support_rate": valid,
                    "alice_local_decode_rate": bob_rate,
                    "bob_local_decode_rate": alice_rate,
                    "bidirectional_decode_rate": both,
                    "bidirectional_wilson_low": low,
                    "bidirectional_wilson_high": high,
                    "metric_scope": "single six-qubit block gate/readout simulation",
                }
            )

    paths = {
        "circuits": RAW / "c_quantum_circuit_inventory_release_v7.csv",
        "exact": RAW / "c_quantum_exact_32_release_v7.csv",
        "branches": RAW / "c_quantum_v7_128_branches_release_v7.csv",
        "ideal_shots": RAW / "c_quantum_ideal_shots_release_v7.csv",
        "channels": RAW / "c_quantum_channel_noise_release_v7.csv",
        "gate_noise": RAW / "c_quantum_gate_noise_release_v7.csv",
    }
    for name, rows in (
        ("circuits", circuit_rows),
        ("exact", exact_rows),
        ("branches", branch_rows),
        ("ideal_shots", shot_rows),
        ("channels", channel_rows),
        ("gate_noise", gate_rows),
    ):
        write_csv(paths[name], rows)
    return {
        "row_counts": {"circuits": len(circuit_rows), "exact": len(exact_rows), "branches": len(branch_rows), "ideal_shots": len(shot_rows), "channels": len(channel_rows), "gate_noise": len(gate_rows)},
        "hashes": {name: sha256_file(path) for name, path in paths.items()},
        "minimum_preparation_fidelity": min(row["preparation_fidelity"] for row in circuit_rows),
        "minimum_exact_bidirectional_decode": min(row["bidirectional_decode_probability"] for row in exact_rows),
        "successful_v7_branches": sum(bool(row["branch_success"]) for row in branch_rows),
        "maximum_numerical_amplitude_error": max(row["maximum_aligned_amplitude_error"] for row in circuit_rows),
    }


def probability_cache() -> dict[tuple[str, str, str, float], np.ndarray]:
    cache: dict[tuple[str, str, str, float], np.ndarray] = {}
    for cluster, message_a, message_b, p in itertools.product(("phi1", "phi2"), BELL_LABELS, BELL_LABELS, (0.0, 0.02, 0.05, 0.10)):
        ideal = Statevector(analytic_encoded_target(cluster, message_a, message_b))
        density = apply_noise_to_qubits(ideal, "depolarizing", p, (2, 3))
        probabilities = bell_decode_probabilities(density)
        probabilities = np.maximum(probabilities, 0)
        cache[(cluster, message_a, message_b, p)] = probabilities / probabilities.sum()
    return cache


def run_confirmations(session: IAQDSession) -> tuple[str, str]:
    try:
        confirm_b = session.send_confirm(ID_B)
        confirm_a = session.send_confirm(ID_A)
    except ProtocolViolation as error:
        return "not_sent", error.reason
    accepted_b_at_alice = session.deliver(confirm_b, ID_A)
    accepted_a_at_bob = session.deliver(confirm_a, ID_B)
    return (
        "both_accepted" if accepted_b_at_alice.accepted and accepted_a_at_bob.accepted else "rejected",
        ";".join((accepted_b_at_alice.reason, accepted_a_at_bob.reason)),
    )


def run_full_noisy_protocol() -> dict[str, Any]:
    cache = probability_cache()
    rows: list[dict[str, Any]] = []
    failure_examples: list[dict[str, Any]] = []
    modes = (BASIC, RELEASE)
    p_values = (0.0, 0.02, 0.05, 0.10)
    for mode_index, mode in enumerate(modes):
        for p_index, p in enumerate(p_values):
            for seed in SEEDS:
                rng_seed = int(np.random.SeedSequence([seed, mode_index, p_index, 0xC4]).generate_state(1)[0])
                rng = np.random.default_rng(rng_seed)
                for trial in range(500):
                    message_a_parts = [str(rng.choice(BELL_LABELS)) for _ in range(4)]
                    message_b_parts = [str(rng.choice(BELL_LABELS)) for _ in range(4)]
                    message_a = "".join(message_a_parts)
                    message_b = "".join(message_b_parts)
                    detection_events = rng.random(64) < (p / 2)
                    detection_errors = int(np.count_nonzero(detection_events))
                    fixtures = []
                    recovered_a_parts: list[str | None] = []
                    recovered_b_parts: list[str | None] = []
                    observed_labels = []
                    for block in range(4):
                        cluster = "phi1" if block % 2 == 0 else "phi2"
                        ma, mb = message_a_parts[block], message_b_parts[block]
                        index = int(rng.choice(64, p=cache[(cluster, ma, mb, p)]))
                        observed = labels_from_count_key(format(index, "06b"))
                        recovered_a, recovered_b, _ = recover_messages(cluster, observed, known_alice=ma, known_bob=mb)
                        recovered_a_parts.append(recovered_a)
                        recovered_b_parts.append(recovered_b)
                        observed_labels.append("/".join(observed))
                        fixtures.append(branch_fixture(cluster, ma, mb, observed))
                    secret_seed = int(np.random.SeedSequence([seed, mode_index, p_index, trial, 0x517]).generate_state(1)[0])
                    with deterministic_protocol_secrets(secret_seed):
                        session = IAQDSession(
                            mode=mode,
                            message_a=message_a,
                            message_b=message_b,
                            lambda_bits=3 if mode == RELEASE else None,
                            ell=32,
                            qber_threshold_num=11,
                            qber_threshold_den=100,
                            quantum_blocks=fixtures,
                        )
                        prelude_pass = session.run_prelude(detection_errors=detection_errors, detection_trials=64)
                        confirm_status = "not_reached"
                        confirm_reason = "detection_failed" if not prelude_pass else ""
                        decode_valid = all(value is not None for value in recovered_a_parts + recovered_b_parts)
                        recovered_a_message = "".join(value or "" for value in recovered_a_parts)
                        recovered_b_message = "".join(value or "" for value in recovered_b_parts)
                        if prelude_pass:
                            if mode == RELEASE:
                                finish_release(session)
                            if not decode_valid:
                                if any(value is None for value in recovered_b_parts):
                                    session.local_abort(ID_A, "QUANTUM_DECODE_INVALID")
                                if any(value is None for value in recovered_a_parts):
                                    session.local_abort(ID_B, "QUANTUM_DECODE_INVALID")
                                confirm_reason = "invalid_truth_table_decode"
                            else:
                                # This is the explicit test-driver seam described in the
                                # manifest: actual Qiskit-derived local results replace the
                                # ideal fixture result before the real I8 validator runs.
                                session.alice.recovered_peer_message = recovered_b_message
                                session.bob.recovered_peer_message = recovered_a_message
                                session._event("test_driver", "NOISY_LOCAL_RECOVERY_INJECTED", result="injected", secret_values_logged=False)
                                confirm_status, confirm_reason = run_confirmations(session)
                        result = session.result()
                    core_both_correct = decode_valid and recovered_a_message == message_a and recovered_b_message == message_b
                    failure_class = (
                        "success"
                        if result.success
                        else "detection_reject"
                        if not prelude_pass
                        else "invalid_local_decode"
                        if not decode_valid
                        else "i8_recovery_mismatch"
                    )
                    row = {
                        "mode": mode,
                        "channel": "transmission_depolarizing_S3_S4",
                        "p": p,
                        "decoy_error_probability": p / 2,
                        "seed": seed,
                        "derived_rng_seed": rng_seed,
                        "trial": trial,
                        "n_blocks": 4,
                        "message_bits_per_party": 8,
                        "lambda_bits": 3 if mode == RELEASE else "",
                        "last_chunk_bits": 2 if mode == RELEASE else "",
                        "ell": 32,
                        "decoy_trials": 64,
                        "threshold_count": math.floor(64 * 0.11),
                        "detection_errors": detection_errors,
                        "detection_pass": prelude_pass,
                        "bell_observations": "|".join(observed_labels),
                        "local_decode_valid": decode_valid,
                        "core_both_messages_correct": core_both_correct,
                        "confirmation_status": confirm_status,
                        "confirmation_reason": confirm_reason,
                        "protocol_success": result.success,
                        "terminal_state": result.terminal_state,
                        "alice_terminal_reason": result.alice_state["terminal_reason"],
                        "bob_terminal_reason": result.bob_state["terminal_reason"],
                        "alice_confirm_accepted": result.alice_state["peer_confirm_accepted"],
                        "bob_confirm_accepted": result.bob_state["peer_confirm_accepted"],
                        "alice_S": result.alice_state["S"],
                        "alice_V": result.alice_state["V"],
                        "bob_S": result.bob_state["S"],
                        "bob_V": result.bob_state["V"],
                        "failure_class": failure_class,
                        "scope": "reference composition at quantum-classical seam; not QDist proof or hardware claim",
                    }
                    rows.append(row)
                    if not result.success and len([item for item in failure_examples if item["mode"] == mode and item["p"] == p and item["failure_class"] == failure_class]) == 0:
                        failure_examples.append({**row, "event_log": session.event_log})

    raw_path = RAW / "c_complete_noisy_protocol_trials_release_v7.csv"
    write_csv(raw_path, rows)
    write_json(RAW / "c_complete_noisy_protocol_failure_examples_release_v7.json", failure_examples)
    aggregates: list[dict[str, Any]] = []
    for mode, p in itertools.product(modes, p_values):
        subset = [row for row in rows if row["mode"] == mode and row["p"] == p]
        total = len(subset)
        for metric, field in (("detection_acceptance", "detection_pass"), ("core_bidirectional_correct", "core_both_messages_correct"), ("full_protocol_acceptance", "protocol_success")):
            successes = sum(bool(row[field]) for row in subset)
            low, high = wilson_interval(successes, total)
            aggregates.append(
                {
                    "mode": mode,
                    "p": p,
                    "metric": metric,
                    "successes": successes,
                    "trials": total,
                    "estimate": successes / total,
                    "wilson_low": low,
                    "wilson_high": high,
                    "zero_count_upper_95": clopper_pearson_upper_zero(total) if successes == 0 else "",
                    "failure_count": total - successes,
                    "scope": "reference composition; finite independent trials",
                }
            )
    aggregate_path = DERIVED / "c_complete_noisy_protocol_aggregate_release_v7.csv"
    write_csv(aggregate_path, aggregates)
    return {
        "trial_rows": len(rows),
        "aggregate_rows": len(aggregates),
        "failure_examples": len(failure_examples),
        "failure_class_counts": {name: sum(row["failure_class"] == name for row in rows) for name in sorted({str(row["failure_class"]) for row in rows})},
        "raw_sha256": sha256_file(raw_path),
        "aggregate_sha256": sha256_file(aggregate_path),
        "boundary": "The production V7 quantum adapter remains ideal-algebra checked; this driver explicitly composes Qiskit outcomes at the documented local-recovery seam.",
    }


def main() -> int:
    summary = {"quantum_core": run_core(), "complete_noisy_protocol": run_full_noisy_protocol()}
    write_json(DERIVED / "c_quantum_summary_release_v7.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
