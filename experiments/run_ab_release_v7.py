"""A/B experiments: original-AQD deterministic witnesses and IMR/EM audits."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

from release_v7_common import (  # noqa: E402
    DERIVED,
    LOGS,
    RAW,
    ROOT,
    SEEDS,
    clopper_pearson_upper_zero,
    historical_checksum_status,
    read_csv,
    run_prism,
    sha256_file,
    wilson_interval,
    write_csv,
    write_json,
)
from run_e1_v6 import (  # noqa: E402
    dictionary_timing,
    shifted_zipf_offset,
    shifted_zipf_recovery,
    shortest_abort_trace,
)
import run_e2_v6 as e2  # noqa: E402
import run_e3_v6 as e3  # noqa: E402


def run_a_original_aqd() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    messages = [b"AQD-controlled-message-A", b"AQD-controlled-message-B"]
    digests = [hashlib.sha256(message).hexdigest() for message in messages]
    if len(set(digests)) != 2:
        raise AssertionError("chosen message collision")
    for chosen_bit, digest in enumerate(digests):
        rows.append(
            {
                "component": "chosen_message",
                "distribution": "two_equal_length_messages",
                "min_entropy_bits": 1,
                "q": 2,
                "recovery_rate": 1,
                "chosen_bit": chosen_bit,
                "guessed_bit": digests.index(digest),
                "privacy_advantage": 0.5,
                "session": "challenge",
                "digest": digest,
                "method": "deterministic lookup; no Monte Carlo",
                "collision_precondition": "the two candidate digests are distinct",
            }
        )
    repeated = b"AQD-repeated-cross-session-message"
    other = b"AQD-different-cross-session-message-"
    equality = []
    for session, message in (("sid-A", repeated), ("sid-B", repeated), ("sid-C", other)):
        digest = hashlib.sha256(message).hexdigest()
        equality.append(digest)
        rows.append(
            {
                "component": "cross_session_equality",
                "distribution": "deterministic",
                "session": session,
                "digest": digest,
                "method": "deterministic hash equality; no Monte Carlo",
                "collision_precondition": "same-message equality is exact; different-message distinction holds for this witness only",
            }
        )
    if not (equality[0] == equality[1] and equality[0] != equality[2]):
        raise AssertionError("cross-session equality witness failed")

    exponent = 1.2
    fractions = (0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0)
    for entropy in (4, 8, 12, 16, 20):
        support = 2**entropy
        q_values = sorted({1, support, *(max(1, min(support, math.ceil(support * f))) for f in fractions)})
        timing_q = min(support, 65536)
        preprocessing, query = dictionary_timing(timing_q)
        for distribution in ("uniform", "zipf"):
            offset = shifted_zipf_offset(entropy, exponent) if distribution == "zipf" else math.nan
            for q in q_values:
                recovery = q / support if distribution == "uniform" else shifted_zipf_recovery(q, offset, exponent)
                rows.append(
                    {
                        "component": "dictionary",
                        "distribution": distribution,
                        "min_entropy_bits": entropy,
                        "q": q,
                        "recovery_rate": recovery,
                        "zipf_exponent": exponent if distribution == "zipf" else "",
                        "zipf_offset": offset if distribution == "zipf" else "",
                        "preprocessing_seconds": preprocessing if q == timing_q else "",
                        "online_query_seconds": query if q == timing_q else "",
                        "method": "exact distribution sum plus one synthetic dictionary timing",
                        "collision_precondition": "recovery equality assumes no digest collision inside the enumerated dictionary",
                    }
                )
    out = RAW / "a_original_aqd_hash_dictionary_release_v7.csv"
    write_csv(out, rows)

    prism_terminal = run_prism(
        ROOT / "prism" / "original_aqd_abort_terminal_v6.nm",
        ROOT / "prism" / "original_aqd_abort_terminal_v6.props",
        constants=None,
        log_path=LOGS / "a_original_aqd_abort_terminal.log",
    )
    prism_no_timeout = run_prism(
        ROOT / "prism" / "original_aqd_abort_no_timeout_v6.nm",
        ROOT / "prism" / "original_aqd_abort_no_timeout_v6.props",
        constants=None,
        log_path=LOGS / "a_original_aqd_abort_no_timeout.log",
    )
    trace = {
        "scope": "original AQD deterministic early-abort witness; not revised-protocol fairness",
        "terminal_timeout_enabled": shortest_abort_trace(timeout_enabled=True),
        "nonterminal_no_timeout": shortest_abort_trace(timeout_enabled=False),
        "prism_terminal": prism_terminal,
        "prism_no_timeout": prism_no_timeout,
    }
    trace_path = RAW / "a_original_aqd_abort_release_v7.json"
    write_json(trace_path, trace)
    return {
        "hash_rows": len(rows),
        "hash_sha256": sha256_file(out),
        "trace_sha256": sha256_file(trace_path),
        "terminal_trace_steps": len(trace["terminal_timeout_enabled"]) - 1,
        "no_timeout_trace_steps": len(trace["nonterminal_no_timeout"]) - 1,
        "terminal_prism_results": prism_terminal["results"],
        "no_timeout_prism_results": prism_no_timeout["results"],
    }


def audit_history() -> list[dict[str, object]]:
    names = [
        "e1_hash_dictionary_cross_session_v6.csv",
        "e1_abort_terminal_mdp_v6.json",
        "e2_imr_event_level_by_seed_v6.csv",
        "e2_imr_single_particle_qiskit_v6.csv",
        "e2_imr_event_level_aggregate_v6.csv",
        "e3_probe_density_qiskit_crosscheck_v6.csv",
        "e3_probe_qiskit_shots_v6.csv",
    ]
    rows: list[dict[str, object]] = []
    for name in names:
        candidates = [ROOT / "results" / "raw" / name, ROOT / "results" / "derived" / name]
        path = next((item for item in candidates if item.exists()), None)
        if path is None:
            rows.append({"path": name, "status": "MISSING_CURRENT_WORKSPACE"})
        else:
            rows.append(historical_checksum_status(path))
    return rows


def run_b_imr() -> dict[str, object]:
    alice_z = 0.5
    eve_bias = 0.8
    historical_raw = read_csv(ROOT / "results" / "raw" / "e2_imr_event_level_by_seed_v6.csv")
    historical_aggregate = read_csv(ROOT / "results" / "derived" / "e2_imr_event_level_aggregate_v6.csv")
    historic_formula_errors = []
    for row in historical_raw:
        analytic = e2.analytic_undetected(
            row["strategy"], int(row["L"]), float(row["attack_fraction"]), alice_z, eve_bias
        )
        historic_formula_errors.append(abs(float(row["analytic_undetected"]) - analytic))
    for row in historical_aggregate:
        analytic = e2.analytic_undetected(
            row["strategy"], int(row["L"]), float(row["attack_fraction"]), alice_z, eve_bias
        )
        historic_formula_errors.append(abs(float(row["analytic_undetected"]) - analytic))

    raw_rows: list[dict[str, object]] = []
    aggregate_rows: list[dict[str, object]] = []
    strategies = list(e2.STRATEGIES)
    for strategy, length, fraction in itertools.product(strategies, (8, 16, 64), (0.25, 1.0)):
        analytic = e2.analytic_undetected(strategy, length, fraction, alice_z, eve_bias)
        total_success = total_trials = total_attacked = total_attacked_pass = 0
        for base_seed in SEEDS[:5]:
            seed_used = int(np.random.SeedSequence([base_seed, strategies.index(strategy), length, int(fraction * 1000)]).generate_state(1)[0])
            result = e2.simulate_seed(seed_used, 20000, 2000, length, fraction, strategy, alice_z, eve_bias)
            total_success += result["undetected_trials"]
            total_trials += 20000
            total_attacked += result["attacked_particles"]
            total_attacked_pass += result["attacked_particle_passes"]
            raw_rows.append(
                {
                    "strategy": strategy,
                    "L": length,
                    "attack_fraction": fraction,
                    "base_seed": base_seed,
                    "derived_seed": seed_used,
                    "trials": 20000,
                    "undetected": result["undetected_trials"],
                    "attacked_particles": result["attacked_particles"],
                    "attacked_particle_passes": result["attacked_particle_passes"],
                    "analytic_undetected": analytic,
                    "simulation_level": "explicit_BB84_particle_events",
                }
            )
        low, high = wilson_interval(total_success, total_trials)
        aggregate_rows.append(
            {
                "strategy": strategy,
                "L": length,
                "attack_fraction": fraction,
                "seeds": 5,
                "trials": total_trials,
                "undetected": total_success,
                "empirical_undetected": total_success / total_trials,
                "wilson_low": low,
                "wilson_high": high,
                "zero_count_upper_95": clopper_pearson_upper_zero(total_trials) if total_success == 0 else "",
                "analytic_undetected": analytic,
                "absolute_error": abs(total_success / total_trials - analytic),
                "attacked_particle_pass_rate": total_attacked_pass / total_attacked,
            }
        )
    raw_path = RAW / "b_imr_event_regression_by_seed_release_v7.csv"
    agg_path = DERIVED / "b_imr_event_regression_aggregate_release_v7.csv"
    write_csv(raw_path, raw_rows)
    write_csv(agg_path, aggregate_rows)

    previous_seeds = e2.seeds
    try:
        e2.seeds = lambda: [SEEDS[0]]  # type: ignore[assignment]
        qiskit_rows = e2.qiskit_single_particle_rows(8192)
    finally:
        e2.seeds = previous_seeds  # type: ignore[assignment]
    qiskit_path = RAW / "b_imr_single_particle_qiskit_release_v7.csv"
    write_csv(qiskit_path, qiskit_rows)
    return {
        "historical_rows_checked": len(historical_raw) + len(historical_aggregate),
        "historical_max_formula_error": max(historic_formula_errors, default=math.nan),
        "regression_raw_rows": len(raw_rows),
        "regression_configurations": len(aggregate_rows),
        "regression_max_absolute_error": max(float(row["absolute_error"]) for row in aggregate_rows),
        "qiskit_rows": len(qiskit_rows),
        "raw_sha256": sha256_file(raw_path),
        "aggregate_sha256": sha256_file(agg_path),
        "qiskit_sha256": sha256_file(qiskit_path),
    }


def qiskit_shot_qber_seed(theta: float, shots: int, seed: int) -> tuple[int, int]:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator

    circuits = []
    wanted = []
    for label in e3.BB84:
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
        wanted.append("0" if label in {"0", "+"} else "1")
    result = AerSimulator().run(circuits, shots=shots, seed_simulator=seed).result()
    errors = sum(shots - int(result.get_counts(index).get(bit, 0)) for index, bit in enumerate(wanted))
    return errors, shots * 4


def run_b_em() -> dict[str, object]:
    historical = read_csv(ROOT / "results" / "raw" / "e3_probe_density_qiskit_crosscheck_v6.csv")
    historic_errors: list[float] = []
    for row in historical:
        theta = float(row["theta"])
        density = e3.density_metrics(theta)
        qber = (1 - math.cos(theta)) / 4
        distance = math.sin(theta)
        historic_errors += [
            abs(float(row["qber_density"]) - density["qber_density"]),
            abs(float(row["trace_distance_nuclear_norm"]) - density["trace_distance_nuclear_norm"]),
            abs(float(row["qber_analytic"]) - qber),
            abs(float(row["trace_distance_analytic"]) - distance),
        ]

    deterministic_rows: list[dict[str, object]] = []
    for theta in np.linspace(0.0, math.pi / 2, 101):
        density = e3.density_metrics(float(theta))
        qiskit = e3.qiskit_metrics(float(theta))
        qber = (1 - math.cos(float(theta))) / 4
        distance = math.sin(float(theta))
        guess = (1 + distance) / 2
        component_error = max(
            density["unitarity_error"],
            density["bb84_evolution_error"],
            abs(density["qber_density"] - qber),
            abs(density["trace_distance_nuclear_norm"] - distance),
            abs(density["helstrom_guess"] - guess),
            abs(qiskit["qber_qiskit"] - qber),
            abs(qiskit["trace_distance_qiskit"] - distance),
        )
        deterministic_rows.append(
            {
                "theta": float(theta),
                **density,
                **qiskit,
                "qber_analytic": qber,
                "trace_distance_analytic": distance,
                "helstrom_analytic": guess,
                "maximum_component_error": component_error,
                "scope": "specified_U_theta_probe_family_only",
            }
        )
    density_path = RAW / "b_em_density_qiskit_101_release_v7.csv"
    write_csv(density_path, deterministic_rows)

    shot_rows: list[dict[str, object]] = []
    for theta in (0.0, math.pi / 8, math.pi / 4, 3 * math.pi / 8, math.pi / 2):
        for seed in SEEDS[:5]:
            errors, total = qiskit_shot_qber_seed(theta, 8192, seed)
            low, high = wilson_interval(errors, total)
            shot_rows.append(
                {
                    "theta": theta,
                    "seed": seed,
                    "shots_per_bb84_state": 8192,
                    "total_shots": total,
                    "errors": errors,
                    "qber_empirical": errors / total,
                    "qber_wilson_low": low,
                    "qber_wilson_high": high,
                    "qber_analytic": (1 - math.cos(theta)) / 4,
                    "scope": "specified_U_theta_probe_family_only",
                }
            )
    shot_path = RAW / "b_em_qiskit_shots_by_seed_release_v7.csv"
    write_csv(shot_path, shot_rows)
    return {
        "historical_rows_checked": len(historical),
        "historical_recompute_max_abs_error": max(historic_errors, default=math.nan),
        "deterministic_rows": len(deterministic_rows),
        "deterministic_max_component_error": max(float(row["maximum_component_error"]) for row in deterministic_rows),
        "shot_rows": len(shot_rows),
        "density_sha256": sha256_file(density_path),
        "shots_sha256": sha256_file(shot_path),
    }


def main() -> int:
    history = audit_history()
    history_path = DERIVED / "ab_historical_checksum_audit_release_v7.csv"
    write_csv(history_path, history)
    summary = {
        "scope": "A original-AQD deterministic regression plus B IMR/EM evidence audit",
        "historical_checksum_audit": {
            "rows": len(history),
            "status_counts": {status: sum(row.get("status") == status for row in history) for status in sorted({str(row.get("status")) for row in history})},
            "sha256": sha256_file(history_path),
        },
        "A": run_a_original_aqd(),
        "B_IMR": run_b_imr(),
        "B_EM": run_b_em(),
    }
    summary["reuse_decision"] = {
        "E1": "verified_reuse_candidate_plus_new_regression" if all("MATCH" in str(row.get("status")) for row in history[:2]) else "pending",
        "E2": "verified_reuse_candidate_plus_new_regression" if all("MATCH" in str(row.get("status")) for row in history[2:5]) and summary["B_IMR"]["historical_max_formula_error"] <= 1e-15 else "pending",
        "E3": "verified_reuse_candidate_plus_new_regression" if all("MATCH" in str(row.get("status")) for row in history[5:]) and summary["B_EM"]["historical_recompute_max_abs_error"] <= 1e-12 else "pending",
    }
    write_json(DERIVED / "ab_summary_release_v7.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
