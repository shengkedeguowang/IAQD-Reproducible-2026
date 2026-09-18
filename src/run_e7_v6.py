from __future__ import annotations

import json
import math
import statistics
import time
import tracemalloc
from typing import Any

import numpy as np
from qiskit import transpile
from qiskit.quantum_info import Statevector
from scipy.stats import binom

from cluster_analytic_v6 import analytic_cluster_target
from cluster_circuit_v6 import explicit_cluster_circuit, full_protocol_circuit
from common_v6 import PROCESSED, RAW, TABLES, runtime_config, seeds, sha256_file, summary_row, write_csv
from protocol_v6 import IAQDReferenceExecutor


def ceil_log2_binomial(total: int, chosen: int) -> int:
    value = math.comb(total, chosen)
    return 0 if value <= 1 else (value - 1).bit_length()


def theoretical_compact_decoy_bits(n_blocks: int, ell: int) -> int:
    return 2 * ceil_log2_binomial(n_blocks + ell, ell) + 4 * ell


def threshold_decoy_search() -> list[dict[str, Any]]:
    config = runtime_config()
    e7 = config["e7"]
    maximum = int(e7["decoy_search_max_L"])
    target_false_abort = float(e7["target_honest_false_abort"])
    target_miss = float(e7["target_attack_miss"])
    q1 = float(config["e6"]["attack_total_qber"])
    q0_values = [float(value) for value in config["e6"]["honest_qber_values"]]
    qth_values = [float(value) for value in config["e6"]["qber_threshold_values"]]
    lengths = np.arange(1, maximum + 1)
    rows: list[dict[str, Any]] = []
    for q0 in q0_values:
        for qth in qth_values:
            thresholds = np.floor(lengths * qth)
            honest_accept = binom.cdf(thresholds, lengths, q0)
            attack_accept = binom.cdf(thresholds, lengths, q1)
            valid = (1 - honest_accept <= target_false_abort) & (attack_accept <= target_miss)
            indexes = np.flatnonzero(valid)
            if len(indexes):
                index = int(indexes[0])
                minimum_total = int(lengths[index])
                ell = math.ceil(minimum_total / 2)
                effective_total = 2 * ell
                effective_threshold = math.floor(effective_total * qth)
                rows.append(
                    {
                        "scenario": "threshold_derived",
                        "q0": q0,
                        "q1": q1,
                        "qth": qth,
                        "target_honest_false_abort": target_false_abort,
                        "target_attack_miss": target_miss,
                        "minimum_total_decoys_L": minimum_total,
                        "ell_per_sequence": ell,
                        "effective_total_decoys_2ell": effective_total,
                        "honest_false_abort_at_minimum_L": float(1 - honest_accept[index]),
                        "attack_miss_at_minimum_L": float(attack_accept[index]),
                        "honest_false_abort_at_effective_2ell": float(1 - binom.cdf(effective_threshold, effective_total, q0)),
                        "attack_miss_at_effective_2ell": float(binom.cdf(effective_threshold, effective_total, q1)),
                        "search_max_L": maximum,
                        "status": "found",
                    }
                )
            else:
                rows.append(
                    {
                        "scenario": "threshold_derived",
                        "q0": q0,
                        "q1": q1,
                        "qth": qth,
                        "target_honest_false_abort": target_false_abort,
                        "target_attack_miss": target_miss,
                        "minimum_total_decoys_L": "",
                        "ell_per_sequence": "",
                        "effective_total_decoys_2ell": "",
                        "honest_false_abort_at_minimum_L": "",
                        "attack_miss_at_minimum_L": "",
                        "honest_false_abort_at_effective_2ell": "",
                        "attack_miss_at_effective_2ell": "",
                        "search_max_L": maximum,
                        "status": "not_found",
                    }
                )
    return rows


def ell_scenarios(ideal_ell: int, threshold_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {"ell_scenario": "reference_executor_small", "ell": 4, "q0": "", "q1": "", "qth": "", "source": "V6 reference-executor benchmark"},
        {
            "ell_scenario": "ideal_zero_background_full_random_basis_IMR",
            "ell": ideal_ell,
            "q0": 0.0,
            "q1": 0.25,
            "qth": 0.0,
            "source": "analytic kappa=128 ideal full-random-basis IMR; not deployment parameter",
        },
    ]
    for item in threshold_rows:
        if item["status"] != "found":
            continue
        rows.append(
            {
                "ell_scenario": f"threshold_q0_{item['q0']}_qth_{item['qth']}",
                "ell": int(item["ell_per_sequence"]),
                "q0": item["q0"],
                "q1": item["q1"],
                "qth": item["qth"],
                "source": "exact binomial-tail threshold search",
            }
        )
    return rows


def paper_quantum_resources(scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
    n_values = [int(value) for value in runtime_config()["e7"]["performance_n_blocks"]]
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        ell = int(scenario["ell"])
        for n_blocks in n_values:
            rows.append(
                {
                    "resource_class": "paper_analytic",
                    "ell_scenario": scenario["ell_scenario"],
                    "source": scenario["source"],
                    "n_blocks": n_blocks,
                    "message_length_bits_per_party": 2 * n_blocks,
                    "ell_per_sequence": ell,
                    "q0": scenario["q0"],
                    "q1": scenario["q1"],
                    "qth": scenario["qth"],
                    "Q_prep_qubits": 6 * n_blocks + 2 * ell,
                    "Q_tx_qubits": 2 * n_blocks + 2 * ell,
                    "Q_store_qubits": 4 * n_blocks,
                    "N_meas_measurements": 6 * n_blocks + 2 * ell,
                    "eta_q": (4 * n_blocks) / (6 * n_blocks + 2 * ell),
                    "ideal_random_basis_IMR_P_und": (3 / 4) ** (2 * ell),
                    "theoretical_compact_decoy_bits": theoretical_compact_decoy_bits(n_blocks, ell),
                    "actual_wire_bytes": "not_applicable_paper_analytic",
                    "ell_consistent_within_row": True,
                    "claim_boundary": "analytic finite-parameter resource count; not hardware performance",
                }
            )
    return rows


def _count_ops(circuit: Any) -> dict[str, int]:
    return {str(name): int(count) for name, count in circuit.count_ops().items()}


def _single_qubit_count(ops: dict[str, int]) -> int:
    return sum(count for name, count in ops.items() if name not in {"cx", "cz", "ecr", "measure", "barrier"})


def _two_qubit_count(ops: dict[str, int]) -> int:
    return sum(int(ops.get(name, 0)) for name in ("cx", "cz", "ecr"))


def circuit_inventory() -> list[dict[str, Any]]:
    config = runtime_config()["e7"]
    basis = [str(value) for value in config["transpiler_basis_gates"]]
    transpiler_seed = int(config["transpiler_seed"])
    rows: list[dict[str, Any]] = []
    for initial_state in ("phi1", "phi2"):
        preparation = explicit_cluster_circuit(initial_state)
        full = full_protocol_circuit(initial_state, "11", "11", measure=True)
        prep_ops = _count_ops(preparation)
        full_ops = _count_ops(full)
        compiled = transpile(
            full,
            basis_gates=basis,
            optimization_level=1,
            seed_transpiler=transpiler_seed,
        )
        compiled_ops = _count_ops(compiled)
        fidelity = float(abs(np.vdot(analytic_cluster_target(initial_state), Statevector.from_instruction(preparation).data)) ** 2)
        rows.append(
            {
                "initial_state": initial_state,
                "resource_class": "current_Qiskit_explicit_circuit_not_minimal_gate_claim",
                "preparation_qubits": preparation.num_qubits,
                "preparation_single_qubit_gates": _single_qubit_count(prep_ops),
                "preparation_two_qubit_gates": _two_qubit_count(prep_ops),
                "preparation_depth": preparation.depth(),
                "preparation_count_ops": json.dumps(prep_ops, sort_keys=True),
                "preparation_fidelity": fidelity,
                "encoding_message_A": "11",
                "encoding_message_B": "11",
                "alice_encoding_single_qubit_gates": 2,
                "bob_encoding_single_qubit_gates": 2,
                "bell_basis_single_qubit_gates": 3,
                "bell_basis_two_qubit_gates": 3,
                "full_single_block_single_qubit_gates_total": _single_qubit_count(full_ops),
                "full_single_block_two_qubit_gates_total": _two_qubit_count(full_ops),
                "full_single_block_measurements": int(full_ops.get("measure", 0)),
                "full_single_block_depth": full.depth(),
                "full_single_block_count_ops": json.dumps(full_ops, sort_keys=True),
                "pre_transpile_total_operations": sum(full_ops.values()),
                "post_transpile_single_qubit_gates": _single_qubit_count(compiled_ops),
                "post_transpile_two_qubit_gates": _two_qubit_count(compiled_ops),
                "post_transpile_measurements": int(compiled_ops.get("measure", 0)),
                "post_transpile_total_operations": sum(compiled_ops.values()),
                "post_transpile_depth": compiled.depth(),
                "post_transpile_count_ops": json.dumps(compiled_ops, sort_keys=True),
                "basis_gates": ",".join(basis),
                "transpiler_seed": transpiler_seed,
                "transmitted_data_qubits": 2,
                "local_storage_qubits": 4,
                "joint_6n_state_constructed": False,
                "minimum_gate_claim": False,
            }
        )
    return rows


def paper_classical_resources(scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
    config = runtime_config()["e7"]
    benchmark = config["benchmark_parameter_set"]
    rho = int(benchmark["rho_bits"])
    d_h = int(benchmark["d_H_bits"])
    nu = int(benchmark["nu_bits"])
    h = int(benchmark["h_bits"])
    tau = int(benchmark["tau_bits"])
    commitment = int(benchmark["commitment_bits"])
    sigma = int(benchmark["salt_bits"])
    n_values = [int(value) for value in config["performance_n_blocks"]]
    lambda_values = [int(value) for value in config["performance_lambda_bits"]]
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        ell = int(scenario["ell"])
        for n_blocks in n_values:
            d_dec = theoretical_compact_decoy_bits(n_blocks, ell)
            common = {
                "resource_class": "paper_analytic_benchmark_parameter_set",
                "benchmark_status": benchmark["status"],
                "n_blocks": n_blocks,
                "ell_per_sequence": ell,
                "ell_scenario": scenario["ell_scenario"],
                "rho_bits": rho,
                "d_H_bits": d_h,
                "nu_bits": nu,
                "h_bits": h,
                "tau_bits": tau,
                "commitment_bits": commitment,
                "salt_bits": sigma,
                "theoretical_compact_decoy_bits": d_dec,
                "KE_QKD_communication_included": False,
                "parameter_source": "paper symbolic formula plus explicitly labelled benchmark_parameter_set",
            }
            c_orig = rho + d_dec + 4 * n_blocks + 2 * d_h
            c_base = 2 * nu + 6 * (h + tau) + 4 * n_blocks + d_dec + 2 * math.ceil(math.log2(n_blocks + ell + 1))
            rows.append({**common, "mode": "original_AQD", "lambda_bits": "", "r": "", "formula": "rho+D_dec+4n+2d_H", "paper_analytic_bits": c_orig})
            rows.append({**common, "mode": "IAQD_basic", "lambda_bits": "", "r": "", "formula": "2nu+6(h+tau)+4n+D_dec+2ceil(log2(n+ell+1))", "paper_analytic_bits": c_base})
            for lambda_bits in lambda_values:
                if lambda_bits > 2 * n_blocks:
                    continue
                r = math.ceil(2 * n_blocks / lambda_bits)
                c_fair = c_base + 4 * n_blocks + 2 * r * (commitment + sigma + h + tau) + 2 * (h + tau)
                rows.append(
                    {
                        **common,
                        "mode": "IAQD_fair",
                        "lambda_bits": lambda_bits,
                        "r": r,
                        "formula": "C_base+4n+2r(c+sigma+h+tau)+2(h+tau)",
                        "paper_analytic_bits": c_fair,
                    }
                )
    return rows


def performance_matrix() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    config = runtime_config()["e7"]
    runs = int(config["performance_runs"])
    n_values = [int(value) for value in config["performance_n_blocks"]]
    lambda_values = [int(value) for value in config["performance_lambda_bits"]]
    ell_values = [int(value) for value in config["performance_ell_values"]]
    modes: list[tuple[str, int, int | None]] = [("basic", n, None) for n in n_values]
    modes.extend(("fair", n, value) for n in n_values for value in lambda_values if value <= 2 * n)
    raw_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    for ell in ell_values:
        for mode, n_blocks, lambda_bits in modes:
            message_a = "".join("00" if index % 2 == 0 else "11" for index in range(n_blocks))
            message_b = "".join("11" if index % 2 == 0 else "00" for index in range(n_blocks))
            durations: list[float] = []
            throughputs: list[float] = []
            peaks: list[int] = []
            successes = 0
            metadata: dict[str, Any] | None = None
            for run_index, seed in enumerate(seeds()):
                executor = IAQDReferenceExecutor(seed, decoys_per_sequence=ell)
                tracemalloc.start()
                started = time.perf_counter()
                result = (
                    executor.run_basic(message_a, message_b)
                    if mode == "basic"
                    else executor.run_fair(message_a, message_b, int(lambda_bits), requested_leader="Alice" if run_index % 2 == 0 else "Bob")
                )
                elapsed = time.perf_counter() - started
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                success = result.success and result.terminal_state == "COMPLETED"
                if not success:
                    raise AssertionError("performance reference execution did not complete")
                successes += int(success)
                duration_ms = elapsed * 1000
                throughput = (4 * n_blocks) / elapsed
                durations.append(duration_ms)
                throughputs.append(throughput)
                peaks.append(peak)
                metadata = result.metrics
                raw_rows.append(
                    {
                        "mode": mode,
                        "n_blocks": n_blocks,
                        "message_length_bits_per_party": 2 * n_blocks,
                        "lambda_bits": "" if lambda_bits is None else lambda_bits,
                        "ell_per_sequence": ell,
                        "seed": seed,
                        "runtime_ms": duration_ms,
                        "throughput_bidirectional_message_bits_per_second": throughput,
                        "peak_memory_bytes": peak,
                        "actual_total_wire_bytes": result.metrics["actual_total_wire_bytes"],
                        "actual_decoy_record_bytes": result.metrics["actual_decoy_record_bytes"],
                        "record_count": result.metrics["one_way_authenticated_record_count"],
                        "upper_layer_round_count": result.metrics["upper_layer_round_count"],
                        "terminal_state": result.terminal_state,
                        "honest_success": success,
                        "logical_erasure": result.logical_erasure,
                        "quantum_execution_scope": "independent_six_qubit_blocks; ell affects decoy resources/authentication/serialization only",
                        **result.metrics,
                    }
                )
            assert metadata is not None
            q25, q75 = np.percentile(durations, (25, 75))
            mean = statistics.mean(durations)
            standard_deviation = statistics.stdev(durations)
            half = 1.959963984540054 * standard_deviation / math.sqrt(runs)
            aggregate_rows.append(
                {
                    "mode": mode,
                    "n_blocks": n_blocks,
                    "message_length_bits_per_party": 2 * n_blocks,
                    "lambda_bits": "" if lambda_bits is None else lambda_bits,
                    "ell_per_sequence": ell,
                    "runtime_mean_ms": mean,
                    "runtime_median_ms": statistics.median(durations),
                    "runtime_std_ms": standard_deviation,
                    "runtime_IQR_ms": float(q75 - q25),
                    "runtime_mean_95_CI_low_ms": max(0.0, mean - half),
                    "runtime_mean_95_CI_high_ms": mean + half,
                    "bidirectional_message_throughput_mean_bits_per_second": statistics.mean(throughputs),
                    "peak_memory_max_bytes": max(peaks),
                    "actual_total_wire_bytes": metadata["actual_total_wire_bytes"],
                    "actual_decoy_record_bytes": metadata["actual_decoy_record_bytes"],
                    "record_count": metadata["one_way_authenticated_record_count"],
                    "upper_layer_round_count": metadata["upper_layer_round_count"],
                    "terminal_state": "COMPLETED",
                    "honest_success_rate": successes / runs,
                    "independent_runs": runs,
                    "performance_scope": "Python reference implementation; excludes quantum hardware, optical/network latency, and external KE/QKD",
                    **metadata,
                }
            )
    return raw_rows, aggregate_rows


def authentication_accounting(aggregates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in aggregates:
        n_blocks = int(item["n_blocks"])
        lambda_value = item["lambda_bits"]
        if item["mode"] == "basic":
            r: int | str = ""
            paper_total = 12
            paper_generate = paper_verify = 6
            paper_commitment_operations = 0
            aggregation_assumption = "not_applicable_basic"
        else:
            r = math.ceil(2 * n_blocks / int(lambda_value))
            paper_total = 16 + 4 * r
            paper_generate = paper_verify = 8 + 2 * r
            paper_commitment_operations = 4 * r
            aggregation_assumption = "two aggregated commitment records plus two opening records per release block"
        reference_total = int(item["reference_actual_auth_operation_count"])
        rows.append(
            {
                "mode": item["mode"],
                "n_blocks": n_blocks,
                "lambda_bits": lambda_value,
                "ell_per_sequence": item["ell_per_sequence"],
                "r": r,
                "paper_logical_auth_generation_count": paper_generate,
                "paper_logical_auth_verification_count": paper_verify,
                "paper_logical_auth_operation_count": paper_total,
                "paper_commitment_generation_or_verification_count": paper_commitment_operations,
                "paper_record_aggregation_assumption": aggregation_assumption,
                "reference_actual_auth_generation_count": item["reference_actual_auth_generation_count"],
                "reference_actual_auth_verification_count": item["reference_actual_auth_verification_count"],
                "reference_actual_auth_operation_count": reference_total,
                "reference_commitment_record_mac_generation_count": item["commitment_record_mac_generation_count"],
                "reference_commitment_record_mac_verification_count": item["commitment_record_mac_verification_count"],
                "reference_opening_record_mac_generation_count": item["opening_record_mac_generation_count"],
                "reference_opening_record_mac_verification_count": item["opening_record_mac_verification_count"],
                "reference_engineering_outer_mac_generation_count": item["engineering_outer_record_mac_generation_count"],
                "reference_engineering_outer_mac_verification_count": item["engineering_outer_record_mac_verification_count"],
                "reference_minus_paper_operations": reference_total - paper_total,
                "difference_reason": "reference uses per-block commitment records and engineering outer MACs; paper formula assumes aggregated commitments",
            }
        )
    return rows


def reference_resources(aggregates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in aggregates:
        n_blocks = int(item["n_blocks"])
        ell = int(item["ell_per_sequence"])
        compact = theoretical_compact_decoy_bits(n_blocks, ell)
        rows.append(
            {
                "resource_class": "reference_implementation_actual",
                "mode": item["mode"],
                "n_blocks": n_blocks,
                "lambda_bits": item["lambda_bits"],
                "ell_per_sequence": ell,
                "prepared_qubits_configured": item["prepared_qubits"],
                "transmitted_qubits_configured": item["transmitted_qubits"],
                "local_storage_qubits_configured": item["local_storage_qubits"],
                "protocol_total_measurements": item["protocol_total_measurement_count"],
                "theoretical_compact_decoy_bits": compact,
                "actual_decoy_payload_bytes": item["actual_decoy_payload_bytes"],
                "actual_decoy_record_bytes": item["actual_decoy_record_bytes"],
                "actual_total_wire_bytes": item["actual_total_wire_bytes"],
                "encoding_overhead_ratio_payload_bits_over_compact": (8 * int(item["actual_decoy_payload_bytes"])) / compact,
                "actual_header_bytes": item["header_bytes"],
                "actual_tag_bytes": item["tag_bytes"],
                "record_count": item["record_count"],
                "upper_layer_round_count": item["upper_layer_round_count"],
                "KE_internal_round_count": item["KE_internal_round_count"],
                "ell_consistent_within_row": int(item["prepared_qubits"]) == 6 * n_blocks + 2 * ell,
                "full_D_decoy_in_wire_record": True,
                "tau_dist_covers_full_D_decoy": True,
            }
        )
    return rows


def _write_tables(circuits: list[dict[str, Any]], resources: list[dict[str, Any]], auth: list[dict[str, Any]]) -> None:
    circuit_lines = [
        r"\begin{tabular}{lrrrrrrrr}",
        r"\hline",
        r"State & Prep $G_1$ & Prep $G_2$ & Prep depth & Full $G_1$ & Full $G_2$ & Meas. & Full depth \\",
        r"\hline",
    ]
    for row in circuits:
        circuit_lines.append(
            f"{row['initial_state']} & {row['preparation_single_qubit_gates']} & {row['preparation_two_qubit_gates']} & "
            f"{row['preparation_depth']} & {row['full_single_block_single_qubit_gates_total']} & "
            f"{row['full_single_block_two_qubit_gates_total']} & {row['full_single_block_measurements']} & {row['full_single_block_depth']} \\\\" 
        )
    circuit_lines.extend([r"\hline", r"\end{tabular}"])
    (TABLES / "e7_circuit_resources_v6.tex").write_text("\n".join(circuit_lines) + "\n", encoding="utf-8")

    selected = [row for row in resources if int(row["n_blocks"]) == 32 and (row["mode"] == "basic" or str(row["lambda_bits"]) == "4")]
    resource_lines = [
        r"\begin{tabular}{llrrrr}",
        r"\hline",
        r"Mode & $(\ell,\lambda)$ & Decoy payload B & Decoy record B & Total wire B & Rounds \\",
        r"\hline",
    ]
    for row in selected:
        resource_lines.append(
            f"{row['mode']} & ({row['ell_per_sequence']},{row['lambda_bits'] or '--'}) & {row['actual_decoy_payload_bytes']} & "
            f"{row['actual_decoy_record_bytes']} & {row['actual_total_wire_bytes']} & {row['upper_layer_round_count']} \\\\" 
        )
    resource_lines.extend([r"\hline", r"\end{tabular}"])
    (TABLES / "e7_wire_resources_v6.tex").write_text("\n".join(resource_lines) + "\n", encoding="utf-8")

    auth_lines = [
        r"\begin{tabular}{llrrr}",
        r"\hline",
        r"Mode & $(n,\lambda,\ell)$ & Paper auth ops & Reference ops & Difference \\",
        r"\hline",
    ]
    for row in auth:
        if int(row["n_blocks"]) == 32 and int(row["ell_per_sequence"]) in {4, 155} and (row["mode"] == "basic" or str(row["lambda_bits"]) == "4"):
            auth_lines.append(
                f"{row['mode']} & ({row['n_blocks']},{row['lambda_bits'] or '--'},{row['ell_per_sequence']}) & "
                f"{row['paper_logical_auth_operation_count']} & {row['reference_actual_auth_operation_count']} & "
                f"{row['reference_minus_paper_operations']} \\\\" 
            )
    auth_lines.extend([r"\hline", r"\end{tabular}"])
    (TABLES / "e7_authentication_resources_v6.tex").write_text("\n".join(auth_lines) + "\n", encoding="utf-8")


def run() -> list[dict[str, object]]:
    config = runtime_config()["e7"]
    ideal_ell = math.ceil(int(config["security_bits"]) / (-2 * math.log2(1 - float(config["f_min_ideal"]) / 4)))
    threshold_rows = threshold_decoy_search()
    decoy_rows = [
        {
            "scenario": "ideal_zero_background_full_random_basis_IMR",
            "security_bits": int(config["security_bits"]),
            "f": float(config["f_min_ideal"]),
            "ell_per_sequence": ideal_ell,
            "total_decoys_2ell": 2 * ideal_ell,
            "analytic_undetected": (3 / 4) ** (2 * ideal_ell),
            "deployment_status": "ideal_analytic_value_not_real_optical_parameter",
        }
    ] + threshold_rows
    decoy_path = RAW / "e7_ideal_and_threshold_decoy_sizes_v6.csv"
    write_csv(decoy_path, decoy_rows)

    scenarios = ell_scenarios(ideal_ell, threshold_rows)
    analytic_quantum = paper_quantum_resources(scenarios)
    analytic_quantum_path = RAW / "e7_paper_analytic_quantum_resources_v6.csv"
    write_csv(analytic_quantum_path, analytic_quantum)

    classic = paper_classical_resources(scenarios)
    classic_path = RAW / "e7_paper_analytic_classical_communication_v6.csv"
    write_csv(classic_path, classic)

    circuits = circuit_inventory()
    circuit_path = RAW / "e7_qiskit_preparation_and_full_circuit_resources_v6.csv"
    write_csv(circuit_path, circuits)

    performance_rows, aggregates = performance_matrix()
    performance_path = RAW / "e7_reference_performance_by_seed_v6.csv"
    aggregate_path = PROCESSED / "e7_reference_performance_aggregate_v6.csv"
    write_csv(performance_path, performance_rows)
    write_csv(aggregate_path, aggregates)

    round_rows: list[dict[str, Any]] = []
    for aggregate in aggregates:
        for item in aggregate["upper_layer_round_schedule"]:
            round_rows.append(
                {
                    "mode": aggregate["mode"],
                    "n_blocks": aggregate["n_blocks"],
                    "lambda_bits": aggregate["lambda_bits"],
                    "ell_per_sequence": aggregate["ell_per_sequence"],
                    "round_index": item["round_index"],
                    "phase": item["phase"],
                    "send_events": "|".join(item["send_events"]),
                    "one_way_authenticated_record_count": aggregate["one_way_authenticated_record_count"],
                    "upper_layer_round_count": aggregate["upper_layer_round_count"],
                    "KE_internal_round_count": aggregate["KE_internal_round_count"],
                }
            )
    round_path = RAW / "e7_upper_layer_round_schedule_v6.csv"
    write_csv(round_path, round_rows)

    reference = reference_resources(aggregates)
    reference_path = RAW / "e7_reference_implementation_resources_v6.csv"
    write_csv(reference_path, reference)

    auth = authentication_accounting(aggregates)
    auth_path = RAW / "e7_paper_vs_reference_authentication_v6.csv"
    write_csv(auth_path, auth)

    external_path = RAW / "e7_external_ke_qkd_scope_v6.csv"
    write_csv(
        external_path,
        [
            {
                "external_primitive": primitive,
                "status": "not_instantiated_external_primitive",
                "quantum_transmissions": "not_measured",
                "classic_messages": "not_measured",
                "internal_round_count": "not_measured",
                "runtime": "not_measured",
                "included_in_IAQD_upper_layer_resources": False,
            }
            for primitive in ("QKD", "authenticated_KE", "PQC-KEM", "quantum_safe_KDF")
        ],
    )

    _write_tables(circuits, reference, auth)
    basic_32_4 = next(row for row in aggregates if row["mode"] == "basic" and row["n_blocks"] == 32 and row["ell_per_sequence"] == 4)
    basic_32_155 = next(row for row in aggregates if row["mode"] == "basic" and row["n_blocks"] == 32 and row["ell_per_sequence"] == 155)
    fair_32_4 = next(row for row in aggregates if row["mode"] == "fair" and row["n_blocks"] == 32 and row["lambda_bits"] == 4 and row["ell_per_sequence"] == 4)
    summaries = [
        summary_row("E7", "ideal_zero_background_full_IMR", "ell_per_sequence", ideal_ell, unit="decoys", raw_hash=sha256_file(decoy_path), notes="Ideal analytic value only; not an optical deployment parameter."),
        summary_row("E7", "threshold_q0_grid", "configurations_with_solution", sum(row["status"] == "found" for row in threshold_rows), unit="configurations", raw_hash=sha256_file(decoy_path)),
        summary_row("E7", "reference_basic_n32_ell4", "actual_total_wire_bytes", basic_32_4["actual_total_wire_bytes"], unit="bytes", raw_hash=sha256_file(reference_path)),
        summary_row("E7", "reference_basic_n32_ell155", "actual_total_wire_bytes", basic_32_155["actual_total_wire_bytes"], unit="bytes", raw_hash=sha256_file(reference_path)),
        summary_row("E7", "explicit_full_protocol_phi1", "two_qubit_gates", next(row["full_single_block_two_qubit_gates_total"] for row in circuits if row["initial_state"] == "phi1"), unit="gates", raw_hash=sha256_file(circuit_path), notes="Current explicit Qiskit implementation; no minimality claim."),
        summary_row("E7", "IAQD_basic", "paper_logical_auth_operations", 12, unit="operations", raw_hash=sha256_file(auth_path)),
        summary_row("E7", "IAQD_basic", "upper_layer_round_count", basic_32_4["upper_layer_round_count"], unit="rounds", raw_hash=sha256_file(round_path)),
        summary_row("E7", "IAQD_fair_n32_lambda4", "upper_layer_round_count", fair_32_4["upper_layer_round_count"], unit="rounds", raw_hash=sha256_file(round_path)),
        summary_row("E7", "reference_basic_n32_ell4", "runtime_median_ms", basic_32_4["runtime_median_ms"], ci_low=basic_32_4["runtime_mean_95_CI_low_ms"], ci_high=basic_32_4["runtime_mean_95_CI_high_ms"], unit="milliseconds", raw_hash=sha256_file(performance_path), notes="Python upper-layer reference implementation; external KE excluded."),
    ]
    write_csv(PROCESSED / "e7_summary_v6.csv", summaries)
    return summaries


if __name__ == "__main__":
    run()
