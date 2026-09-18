from __future__ import annotations

import hashlib
import itertools
from functools import lru_cache
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from common_v6 import PROCESSED, RAW, seeds, sha256_file, summary_row, write_csv
from protocol_v6 import IAQDReferenceExecutor, derive_initial_leader
from bell_truth_table_v6 import branch_fixture, expected_observations, recover_messages
from cluster_circuit_v6 import full_protocol_circuit
from quantum_common_v6 import BELL_LABELS


@lru_cache(maxsize=None)
def actual_bell_support(cluster_type: str, alice: str, bob: str) -> tuple[tuple[str, str, str], ...]:
    state = Statevector.from_instruction(full_protocol_circuit(cluster_type, alice, bob, measure=False))
    decode = QuantumCircuit(6)
    for first, second in ((0, 1), (2, 3), (4, 5)):
        decode.cx(first, second)
        decode.h(first)
    probabilities = np.asarray(state.evolve(decode).probabilities())
    support: list[tuple[str, str, str]] = []
    for index, probability in enumerate(probabilities):
        if float(probability) <= 1e-12:
            continue
        bits = format(index, "06b")[::-1]
        support.append((bits[0:2], bits[2:4], bits[4:6]))
    if set(support) != expected_observations(cluster_type, alice, bob):
        raise AssertionError("multiblock Qiskit support disagrees with Bell truth table")
    return tuple(sorted(support))


def cluster_patterns(n_blocks: int, seed: int) -> dict[str, list[str]]:
    rng = np.random.default_rng(seed + n_blocks)
    return {
        "all_phi1": ["phi1"] * n_blocks,
        "all_phi2": ["phi2"] * n_blocks,
        "alternating": ["phi1" if index % 2 == 0 else "phi2" for index in range(n_blocks)],
        "fixed_seed_random": ["phi1" if int(value) == 0 else "phi2" for value in rng.integers(0, 2, size=n_blocks)],
    }


def message_patterns(n_blocks: int, seed: int, party: str) -> dict[str, str]:
    rng = np.random.default_rng(seed + n_blocks + (0 if party == "Alice" else 1000))
    return {
        "all_00": "00" * n_blocks,
        "all_11": "11" * n_blocks,
        "alternating_00_11": "".join("00" if index % 2 == 0 else "11" for index in range(n_blocks)),
        "fixed_seed_random": "".join(f"{int(value):02b}" for value in rng.integers(0, 4, size=n_blocks)),
    }


def commitment_links_are_exact(result: Any) -> bool:
    for party, mask_name in (("A", "R_A"), ("B", "R_B")):
        reconstructed = "".join(item["chunk"] for item in result.commitments[party])
        if reconstructed != result.masks[mask_name]:
            return False
        if any(len(bytes.fromhex(item["salt"])) < 16 for item in result.commitments[party]):
            return False
    return True


def run_single_block_exact() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    branch_index = 0
    for initial_state, alice_message, bob_message in itertools.product(("phi1", "phi2"), BELL_LABELS, BELL_LABELS):
        for observed in actual_bell_support(initial_state, alice_message, bob_message):
            recovered_alice, recovered_bob, intermediates = recover_messages(
                initial_state, observed, known_alice=alice_message, known_bob=bob_message
            )
            quantum_block = branch_fixture(initial_state, alice_message, bob_message, observed)
            requested_leader = "Alice" if branch_index % 2 == 0 else "Bob"
            result = IAQDReferenceExecutor(seeds()[0] + branch_index).run_fair(
                alice_message,
                bob_message,
                lambda_bits=1,
                requested_leader=requested_leader,
                cluster_sequence=[initial_state],
                quantum_blocks=[quantum_block],
            )
            algebra_event = next(item for item in result.trace if item["event"] == "fair_ciphertexts")
            rows.append(
                {
                    "branch_index": branch_index,
                    "initial_state": initial_state,
                    "alice_message": alice_message,
                    "bob_message": bob_message,
                    "bell_branch_12": observed[0],
                    "bell_branch_34": observed[1],
                    "bell_branch_56": observed[2],
                    "M_B": intermediates["M_B"],
                    "tilde_M_B": intermediates["tilde_M_B"],
                    "M_A": intermediates["M_A"],
                    "tilde_M_A": intermediates["tilde_M_A"],
                    "C_A_hex": algebra_event["C_A"],
                    "C_B_hex": algebra_event["C_B"],
                    "ciphertext_A_formula": "C_A_fair=tilde_M_A XOR K_A_otp XOR R_A",
                    "ciphertext_B_formula": "C_B_fair=tilde_M_B XOR K_B_otp XOR R_B",
                    "decoded_intermediates": str(intermediates),
                    "executor_consumed_bell_labels": "|".join(result.quantum_inputs[0]["bell_labels"]),
                    "recovered_alice_message": recovered_alice,
                    "recovered_bob_message": recovered_bob,
                    "initial_leader": derive_initial_leader(result.sid),
                    "terminal_state": result.terminal_state,
                    "gamma_A_valid": result.gamma_a_valid,
                    "gamma_B_valid": result.gamma_b_valid,
                    "logical_erasure": result.logical_erasure,
                    "commitment_ciphertext_mask_link": commitment_links_are_exact(result),
                    "maximum_information_lead": max(prefix["information_lead"] for prefix in result.fairness_prefixes),
                    "branch_success": (
                        recovered_alice == alice_message
                        and recovered_bob == bob_message
                        and result.success
                        and result.terminal_state == "COMPLETED"
                        and result.logical_erasure
                        and commitment_links_are_exact(result)
                        and algebra_event["M_A"] == intermediates["M_A"]
                        and algebra_event["M_B"] == intermediates["M_B"]
                        and algebra_event["tilde_M_A"] == intermediates["tilde_M_A"]
                        and algebra_event["tilde_M_B"] == intermediates["tilde_M_B"]
                    ),
                }
            )
            branch_index += 1
    if len(rows) != 128 or not all(row["branch_success"] for row in rows):
        raise AssertionError("fair single-block 128-branch matrix failed")
    return rows


def run_multiblock() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    honest_rows: list[dict[str, Any]] = []
    abort_rows: list[dict[str, Any]] = []
    selected_seeds = seeds()[:3]
    configuration_index = 0
    for seed in selected_seeds:
        for n_blocks in (2, 4, 8):
            clusters = cluster_patterns(n_blocks, seed)
            alice_patterns = message_patterns(n_blocks, seed, "Alice")
            bob_patterns = message_patterns(n_blocks, seed, "Bob")
            for cluster_name, cluster_sequence in clusters.items():
                for (alice_name, alice_bits), (bob_name, bob_bits) in itertools.product(alice_patterns.items(), bob_patterns.items()):
                    for lambda_bits in (1, 2, 4):
                        if lambda_bits > 2 * n_blocks:
                            continue
                        for requested_leader in ("Alice", "Bob"):
                            configuration_id = f"fair_{configuration_index:05d}"
                            executor_seed = seed + configuration_index
                            quantum_blocks = []
                            for index, cluster_type in enumerate(cluster_sequence):
                                alice_block = alice_bits[index * 2 : index * 2 + 2]
                                bob_block = bob_bits[index * 2 : index * 2 + 2]
                                observed = actual_bell_support(cluster_type, alice_block, bob_block)[0]
                                block = branch_fixture(cluster_type, alice_block, bob_block, observed)
                                block["source"] = "qiskit_statevector_nonzero_bell_support_cached_per_block"
                                quantum_blocks.append(block)
                            result = IAQDReferenceExecutor(executor_seed).run_fair(
                                alice_bits,
                                bob_bits,
                                lambda_bits=lambda_bits,
                                requested_leader=requested_leader,
                                cluster_sequence=cluster_sequence,
                                quantum_blocks=quantum_blocks,
                            )
                            algebra_event = next(item for item in result.trace if item["event"] == "fair_ciphertexts")
                            max_lead = max(prefix["information_lead"] for prefix in result.fairness_prefixes)
                            last_chunk_a = len(result.commitments["A"][-1]["chunk"])
                            last_chunk_b = len(result.commitments["B"][-1]["chunk"])
                            honest_rows.append(
                                {
                                    "configuration_id": configuration_id,
                                    "seed": seed,
                                    "n_blocks": n_blocks,
                                    "message_length_bits": 2 * n_blocks,
                                    "lambda_bits": lambda_bits,
                                    "cluster_pattern": cluster_name,
                                    "cluster_sequence": "|".join(cluster_sequence),
                                    "executor_cluster_sequence": "|".join(str(block["cluster_type"]) for block in result.quantum_inputs),
                                    "alice_message_pattern": alice_name,
                                    "bob_message_pattern": bob_name,
                                    "alice_message_bits": alice_bits,
                                    "bob_message_bits": bob_bits,
                                    "requested_initial_leader": requested_leader,
                                    "derived_initial_leader": derive_initial_leader(result.sid),
                                    "terminal_state": result.terminal_state,
                                    "success": result.success,
                                    "gamma_after_recovery": next(item["index"] for item in result.trace if item["event"].startswith("fair_keyed_confirmation")) > next(item["index"] for item in result.trace if item["event"] == "fair_message_recovery_complete"),
                                    "commitment_ciphertext_mask_link": commitment_links_are_exact(result),
                                    "block_algebra": __import__("json").dumps(algebra_event["algebra_blocks"], sort_keys=True),
                                    "C_A_hex": algebra_event["C_A"],
                                    "C_B_hex": algebra_event["C_B"],
                                    "ciphertext_A_formula": "C_A_fair=tilde_M_A XOR K_A_otp XOR R_A",
                                    "ciphertext_B_formula": "C_B_fair=tilde_M_B XOR K_B_otp XOR R_B",
                                    "all_block_message_relations_valid": all(
                                        int(item["M_A"], 2) ^ int(item["tilde_M_A"], 2) == int(alice_bits[index * 2:index * 2 + 2], 2)
                                        and int(item["M_B"], 2) ^ int(item["tilde_M_B"], 2) == int(bob_bits[index * 2:index * 2 + 2], 2)
                                        for index, item in enumerate(algebra_event["algebra_blocks"])
                                    ),
                                    "last_chunk_a_bits": last_chunk_a,
                                    "last_chunk_b_bits": last_chunk_b,
                                    "true_last_chunk_accounting": last_chunk_a == min(lambda_bits, (2 * n_blocks) - lambda_bits * (len(result.commitments["A"]) - 1)),
                                    "maximum_information_lead": max_lead,
                                    "fairness_violation": max_lead > lambda_bits,
                                    "joint_6n_state_constructed": False,
                                }
                            )
                            if not result.success or max_lead > lambda_bits or not commitment_links_are_exact(result):
                                raise AssertionError(f"honest fair configuration failed: {configuration_id}")

                            schedule = [str(prefix["sender"]) for prefix in result.fairness_prefixes]
                            for corrupted_party in ("Alice", "Bob"):
                                checked_for_party = 0
                                for prefix_index, next_sender in enumerate(schedule):
                                    if next_sender != corrupted_party:
                                        continue
                                    aborted = IAQDReferenceExecutor(executor_seed).run_fair(
                                        alice_bits,
                                        bob_bits,
                                        lambda_bits=lambda_bits,
                                        requested_leader=requested_leader,
                                        abort_after_openings=prefix_index,
                                        cluster_sequence=cluster_sequence,
                                        quantum_blocks=quantum_blocks,
                                    )
                                    abort_max_lead = max(
                                        (prefix["information_lead"] for prefix in aborted.fairness_prefixes),
                                        default=0,
                                    )
                                    abort_rows.append(
                                        {
                                            "configuration_id": configuration_id,
                                            "corrupted_party": corrupted_party,
                                            "abort_before_opening_prefix": prefix_index,
                                            "next_sender": next_sender,
                                            "terminal_state": aborted.terminal_state,
                                            "openings_completed": len(aborted.fairness_prefixes),
                                            "maximum_information_lead": abort_max_lead,
                                            "lambda_bits": lambda_bits,
                                            "fairness_violation": abort_max_lead > lambda_bits,
                                            "corrupt_party_owns_valid_MAC_key": True,
                                            "prefix_checked": aborted.terminal_state == "ABORT_PARTICIPANT",
                                        }
                                    )
                                    checked_for_party += 1
                                if checked_for_party == 0:
                                    raise AssertionError("a corrupt participant had no allowed abort prefix")
                            configuration_index += 1
    if any(not row["success"] or row["fairness_violation"] for row in honest_rows):
        raise AssertionError("honest multiblock matrix contains a failure")
    if any(not row["prefix_checked"] or row["fairness_violation"] for row in abort_rows):
        raise AssertionError("abort-prefix matrix contains an unresolved result")
    return honest_rows, abort_rows


def run() -> list[dict[str, Any]]:
    exact_rows = run_single_block_exact()
    honest_rows, abort_rows = run_multiblock()
    exact_path = RAW / "fair_single_block_128_exact_branches_v6.csv"
    honest_path = RAW / "fair_multiblock_honest_matrix_v6.csv"
    abort_path = RAW / "fair_corrupt_abort_prefixes_v6.csv"
    write_csv(exact_path, exact_rows)
    write_csv(honest_path, honest_rows)
    write_csv(abort_path, abort_rows)
    alice_configurations = len({row["configuration_id"] for row in abort_rows if row["corrupted_party"] == "Alice"})
    bob_configurations = len({row["configuration_id"] for row in abort_rows if row["corrupted_party"] == "Bob"})
    summaries = [
        summary_row("E6", "fair_single_block", "logical_configurations", 32, unit="configurations", raw_hash=sha256_file(exact_path)),
        summary_row("E6", "fair_single_block", "exact_nonzero_branches", len(exact_rows), unit="branches", raw_hash=sha256_file(exact_path)),
        summary_row("E6", "fair_single_block", "exact_branch_success_rate", sum(row["branch_success"] for row in exact_rows) / len(exact_rows), unit="probability", raw_hash=sha256_file(exact_path)),
        summary_row("E6", "fair_multiblock", "honest_complete_configurations", len(honest_rows), unit="configurations", raw_hash=sha256_file(honest_path)),
        summary_row("E6", "fair_multiblock", "honest_success_rate", sum(row["success"] for row in honest_rows) / len(honest_rows), unit="probability", raw_hash=sha256_file(honest_path)),
        summary_row("E6", "fair_multiblock", "alice_corrupt_configurations", alice_configurations, unit="configurations", raw_hash=sha256_file(abort_path)),
        summary_row("E6", "fair_multiblock", "bob_corrupt_configurations", bob_configurations, unit="configurations", raw_hash=sha256_file(abort_path)),
        summary_row("E6", "fair_multiblock", "abort_prefixes_checked", len(abort_rows), unit="prefixes", raw_hash=sha256_file(abort_path)),
        summary_row("E6", "fair_all", "maximum_information_lead", max([row["maximum_information_lead"] for row in honest_rows] + [row["maximum_information_lead"] for row in abort_rows]), unit="bits", raw_hash=sha256_file(abort_path)),
        summary_row("E6", "fair_all", "fairness_violations", sum(row["fairness_violation"] for row in honest_rows) + sum(row["fairness_violation"] for row in abort_rows), unit="paths", raw_hash=sha256_file(abort_path)),
    ]
    write_csv(PROCESSED / "fair_protocol_summary_v6.csv", summaries)
    return summaries


if __name__ == "__main__":
    run()

