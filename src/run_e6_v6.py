from __future__ import annotations

import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest

from common_v6 import (
    COUNTEREXAMPLES,
    LOGS,
    PRISM_DIR,
    PROCESSED,
    RAW,
    exact_binomial_cdf,
    holm_adjust,
    run_prism,
    runtime_config,
    seeds,
    sha256_file,
    summary_row,
    wilson_interval,
    write_csv,
    write_json,
)


def probability_code(value: float) -> str:
    return f"{int(round(value * 100)):03d}"


def simulate_error_counts(seed_value: int, length: int, q: float, trials: int, batch_trials: int) -> np.ndarray:
    """Generate L Bernoulli error events for every protocol trial."""
    rng = np.random.default_rng(seed_value + length + int(round(q * 10000)))
    counts = np.empty(trials, dtype=np.uint16)
    completed = 0
    while completed < trials:
        current = min(batch_trials, trials - completed)
        particle_errors = rng.random((current, length), dtype=np.float32) < q
        counts[completed : completed + current] = np.sum(particle_errors, axis=1, dtype=np.uint16)
        completed += current
    return counts


def parse_prism_states(path: Path) -> tuple[list[str], dict[int, dict[str, int]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    header = [name.strip() for name in lines[0].strip()[1:-1].split(",")]
    states: dict[int, dict[str, int]] = {}
    for line in lines[1:]:
        if not line.strip():
            continue
        state_id_text, values_text = line.split(":", 1)
        tokens = values_text.strip()[1:-1].split(",")
        values = [1 if value == "true" else 0 if value == "false" else int(value) for value in tokens]
        states[int(state_id_text)] = dict(zip(header, values))
    return header, states


def parse_prism_transitions(path: Path) -> dict[int, list[tuple[int, str]]]:
    adjacency: dict[int, list[tuple[int, str]]] = defaultdict(list)
    lines = path.read_text(encoding="utf-8").splitlines()
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        source = int(parts[0])
        target = int(parts[2])
        action = parts[4] if len(parts) >= 5 else f"choice_{parts[1]}"
        adjacency[source].append((target, action))
    return adjacency


def parse_prism_transition_summary(path: Path) -> dict[str, int]:
    parts = path.read_text(encoding="utf-8").splitlines()[0].split()
    if len(parts) != 3:
        raise ValueError(f"unexpected explicit-transition header in {path}")
    return {"states": int(parts[0]), "choices": int(parts[1]), "transitions": int(parts[2])}


def exact_binomial_test_pvalue(successes: int, trials: int, probability: float) -> float:
    """Handle the p=0/1 degenerate boundary without passing it through SciPy."""
    if probability == 1.0:
        return 1.0 if successes == trials else 0.0
    if probability == 0.0:
        return 1.0 if successes == 0 else 0.0
    return float(binomtest(successes, trials, probability).pvalue)


def shortest_trace_to_states(
    states: dict[int, dict[str, int]],
    adjacency: dict[int, list[tuple[int, str]]],
    targets: set[int],
) -> list[dict[str, Any]]:
    initial = 0
    queue = deque([initial])
    parent: dict[int, tuple[int, str] | None] = {initial: None}
    found: int | None = None
    while queue:
        state_id = queue.popleft()
        if state_id in targets:
            found = state_id
            break
        for target, action in adjacency.get(state_id, []):
            if target not in parent:
                parent[target] = (state_id, action)
                queue.append(target)
    if found is None:
        raise AssertionError("no path to a maximum-lead reachable state")
    reverse_steps: list[tuple[int, str, int]] = []
    current = found
    while parent[current] is not None:
        previous, action = parent[current]  # type: ignore[misc]
        reverse_steps.append((previous, action, current))
        current = previous
    result: list[dict[str, Any]] = [{"state_id": initial, "state": states[initial]}]
    for _, action, target in reversed(reverse_steps):
        result.append({"action": action, "state_id": target, "state": states[target]})
    return result


def run_dtmc() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    config = runtime_config()["e6"]
    trials = int(config["monte_carlo_trials_per_seed"])
    batch_trials = int(config["event_batch_trials"])
    q1 = float(config["attack_total_qber"])
    q0_values = tuple(float(value) for value in config["honest_qber_values"])
    threshold_values = tuple(float(value) for value in config["qber_threshold_values"])
    lengths = tuple(int(value) for value in config["dtmc_lengths"])
    if q0_values != (0.0, 0.01, 0.03, 0.05):
        raise AssertionError("the paper DTMC grid must include q0=0,0.01,0.03,0.05")
    raw_rows: list[dict[str, object]] = []
    aggregate_rows: list[dict[str, object]] = []
    p_values: list[float] = []
    p_refs: list[tuple[dict[str, object], str]] = []
    for length in lengths:
        event_cache: dict[tuple[float, int], np.ndarray] = {}
        for q in (*q0_values, q1):
            for seed_value in seeds():
                event_cache[(q, seed_value)] = simulate_error_counts(seed_value, length, q, trials, batch_trials)
        for q0 in q0_values:
            for threshold in threshold_values:
                threshold_count = math.floor(length * threshold)
                analytic_honest = exact_binomial_cdf(length, q0, threshold)
                analytic_attack = exact_binomial_cdf(length, q1, threshold)
                configuration = f"L{length}_q0{probability_code(q0)}_q1{probability_code(q1)}_t{probability_code(threshold)}"
                honest_total = attack_total = 0
                for seed_value in seeds():
                    honest_accept = int(np.count_nonzero(event_cache[(q0, seed_value)] <= threshold_count))
                    attack_accept = int(np.count_nonzero(event_cache[(q1, seed_value)] <= threshold_count))
                    honest_total += honest_accept
                    attack_total += attack_accept
                    raw_rows.extend(
                        (
                            {
                                "configuration": configuration, "role": "honest", "seed": seed_value,
                                "L": length, "q0": q0, "q1": q1, "q": q0, "qth": threshold, "threshold_count": threshold_count,
                                "trials": trials, "accepted": honest_accept, "acceptance_rate": honest_accept / trials,
                                "simulation_level": "L_independent_error_events_per_trial",
                            },
                            {
                                "configuration": configuration, "role": "attack", "seed": seed_value,
                                "L": length, "q0": q0, "q1": q1, "q": q1, "qth": threshold, "threshold_count": threshold_count,
                                "trials": trials, "accepted": attack_accept, "acceptance_rate": attack_accept / trials,
                                "simulation_level": "L_independent_error_events_per_trial",
                            },
                        )
                    )
                total_trials = trials * len(seeds())
                honest_empirical = honest_total / total_trials
                attack_empirical = attack_total / total_trials
                honest_low, honest_high = wilson_interval(honest_total, total_trials)
                attack_low, attack_high = wilson_interval(attack_total, total_trials)
                if q0 == 0.0 and honest_total != total_trials:
                    raise AssertionError("q0=0 generated a false honest error or abort")
                constants_h = {"L": length, "T": threshold_count, "q": f"{q0:.12g}"}
                constants_a = {"L": length, "T": threshold_count, "q": f"{q1:.12g}"}
                prism_honest = run_prism(
                    PRISM_DIR / "iaqd_detection_v6.pm", PRISM_DIR / "iaqd_detection_v6.props",
                    constants=constants_h, log_path=LOGS / f"e6_dtmc_{configuration}_honest_v6.log", exact=True,
                )
                prism_attack = run_prism(
                    PRISM_DIR / "iaqd_detection_v6.pm", PRISM_DIR / "iaqd_detection_v6.props",
                    constants=constants_a, log_path=LOGS / f"e6_dtmc_{configuration}_attack_v6.log", exact=True,
                )
                honest_p = exact_binomial_test_pvalue(honest_total, total_trials, analytic_honest)
                attack_p = exact_binomial_test_pvalue(attack_total, total_trials, analytic_attack)
                honest_false_abort = 0.0 if q0 == 0.0 else 1 - analytic_honest
                if q0 == 0.0 and (analytic_honest != 1.0 or float(prism_honest["result"]) != 1.0):
                    raise AssertionError("q0=0 analytic and PRISM acceptance must both equal one")
                row: dict[str, object] = {
                    "configuration": configuration,
                    "L": length, "q0": q0, "q1": q1, "qth": threshold, "threshold_count": threshold_count,
                    "honest_acceptance_analytic": analytic_honest,
                    "honest_false_abort_analytic": honest_false_abort,
                    "honest_acceptance_prism": prism_honest["result"],
                    "honest_acceptance_empirical": honest_empirical,
                    "honest_acceptance_ci_low": honest_low,
                    "honest_acceptance_ci_high": honest_high,
                    "attack_miss_analytic": analytic_attack,
                    "attack_miss_prism": prism_attack["result"],
                    "attack_miss_empirical": attack_empirical,
                    "attack_miss_ci_low": attack_low,
                    "attack_miss_ci_high": attack_high,
                    "analytic_prism_max_error": max(abs(float(prism_honest["result"]) - analytic_honest), abs(float(prism_attack["result"]) - analytic_attack)),
                    "monte_carlo_analytic_max_error": max(abs(honest_empirical - analytic_honest), abs(attack_empirical - analytic_attack)),
                    "honest_states": prism_honest["states"], "honest_transitions": prism_honest["transitions"],
                    "attack_states": prism_attack["states"], "attack_transitions": prism_attack["transitions"],
                    "honest_model_construction_seconds": prism_honest["construction_seconds"],
                    "honest_model_checking_seconds": prism_honest["checking_seconds"],
                    "attack_model_construction_seconds": prism_attack["construction_seconds"],
                    "attack_model_checking_seconds": prism_attack["checking_seconds"],
                    "peak_memory_bytes": "not_measured",
                    "honest_prism_log": f"e6_dtmc_{configuration}_honest_v6.log",
                    "attack_prism_log": f"e6_dtmc_{configuration}_attack_v6.log",
                    "honest_p_value": honest_p, "attack_p_value": attack_p,
                    "honest_holm_adjusted_p": "", "attack_holm_adjusted_p": "",
                }
                aggregate_rows.append(row)
                p_values.extend((honest_p, attack_p))
                p_refs.extend(((row, "honest_holm_adjusted_p"), (row, "attack_holm_adjusted_p")))
    for (row, field), adjusted in zip(p_refs, holm_adjust(p_values)):
        row[field] = adjusted
    if len(aggregate_rows) != 48:
        raise AssertionError(f"expected 48 paper DTMC configurations, got {len(aggregate_rows)}")
    return raw_rows, aggregate_rows


def run_fairness() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    config = runtime_config()["e6"]
    rows: list[dict[str, object]] = []
    level_rows: list[dict[str, object]] = []
    model = PRISM_DIR / "iaqd_fair_protocol_v6.nm"
    model_text = model.read_text(encoding="utf-8")
    transition_text = model_text.split("label ", 1)[0]
    forbidden_targets = ("fair_violation", "FairViolation", "maximum_lead", "abs(L_A-L_B)")
    if any(target in transition_text for target in forbidden_targets):
        raise AssertionError("fairness target leaked into PRISM transition logic")

    for corrupted, corrupted_name in ((0, "alice"), (1, "bob")):
        for initial_leader, leader_name in ((0, "alice"), (1, "bob")):
            for n_blocks in config["fairness_n_blocks"]:
                message_length_bits = 2 * int(n_blocks)
                for lambda_bits in config["fairness_lambda_bits"]:
                    if int(lambda_bits) > message_length_bits:
                        continue
                    configuration = f"corrupt_{corrupted_name}_leader_{leader_name}_n{n_blocks}_lambda{lambda_bits}"
                    states_path = COUNTEREXAMPLES / f"e6_{configuration}_all_states_v6.txt"
                    transitions_path = COUNTEREXAMPLES / f"e6_{configuration}_all_transitions_v6.txt"
                    policy_path = COUNTEREXAMPLES / f"e6_{configuration}_reconstructed_witness_policy_v6.json"
                    constants = {
                        "n_blocks": n_blocks,
                        "lambda_bits": lambda_bits,
                        "corrupted": corrupted,
                        "initial_leader": initial_leader,
                    }
                    violation = run_prism(
                        model, PRISM_DIR / "iaqd_fair_protocol_v6.props", prop=1, constants=constants,
                        log_path=LOGS / f"e6_{configuration}_violation_v6.log",
                        states_path=states_path, transitions_path=transitions_path, explicit=True,
                    )
                    terminal_violation = run_prism(
                        model, PRISM_DIR / "iaqd_fair_protocol_v6.props", prop=2, constants=constants,
                        log_path=LOGS / f"e6_{configuration}_terminal_violation_v6.log", explicit=True,
                    )
                    _, states = parse_prism_states(states_path)
                    adjacency = parse_prism_transitions(transitions_path)
                    transition_summary = parse_prism_transition_summary(transitions_path)
                    leads = {state_id: abs(values["L_A"] - values["L_B"]) for state_id, values in states.items()}
                    maximum_lead = max(leads.values())
                    targets = {state_id for state_id, lead in leads.items() if lead == maximum_lead}
                    trace = shortest_trace_to_states(states, adjacency, targets)
                    policy_entries = [
                        {
                            "state_id": int(trace[index]["state_id"]),
                            "action": str(trace[index + 1]["action"]),
                            "target_state_id": int(trace[index + 1]["state_id"]),
                        }
                        for index in range(len(trace) - 1)
                    ]
                    policy_payload = {
                        "kind": "reconstructed_witness_policy",
                        "configuration": configuration,
                        "target_maximum_information_lead": maximum_lead,
                        "entries": policy_entries,
                        "verified_final_state_id": int(trace[-1]["state_id"]),
                        "verified_final_lead": abs(int(trace[-1]["state"]["L_A"]) - int(trace[-1]["state"]["L_B"])),
                    }
                    if not policy_entries or policy_payload["verified_final_lead"] != maximum_lead:
                        raise AssertionError("reconstructed witness policy is empty or misses the max-lead target")
                    write_json(policy_path, policy_payload)
                    aborted_targets = {
                        state_id
                        for state_id, values in states.items()
                        if values.get("phase") == 3 and values.get("termination_reason") == 2
                    }
                    if not aborted_targets:
                        raise AssertionError(f"no reachable participant-abort state for {configuration}")
                    abort_trace = shortest_trace_to_states(states, adjacency, aborted_targets)
                    abort_trace_path = COUNTEREXAMPLES / f"e6_{configuration}_shortest_abort_prefix_v6.json"
                    write_json(
                        abort_trace_path,
                        {
                            "configuration": configuration,
                            "kind": "shortest_participant_abort_prefix",
                            "trace": abort_trace,
                            "terminal_state": abort_trace[-1],
                        },
                    )
                    query_results = [violation, terminal_violation]
                    for k in range(message_length_bits + 1):
                        level_constants = {**constants, "k": k}
                        level = run_prism(
                            model,
                            PRISM_DIR / "iaqd_fair_lead_levels_v6.props",
                            prop=1,
                            constants=level_constants,
                            log_path=LOGS / f"e6_{configuration}_lead_ge_{k:02d}_v6.log",
                            explicit=True,
                        )
                        expected_reachable = maximum_lead >= k
                        if float(level["result"]) != (1.0 if expected_reachable else 0.0):
                            raise AssertionError(f"PRISM/enumeration lead mismatch for {configuration}, k={k}")
                        query_results.append(level)
                        level_rows.append(
                            {
                                "configuration": configuration,
                                "configuration_class": "paper_main",
                                "n_blocks": n_blocks,
                                "message_length_bits": message_length_bits,
                                "lambda_bits": lambda_bits,
                                "corrupted_party": corrupted_name,
                                "initial_leader": leader_name,
                                "k": k,
                                "pmax_reach_abs_lead_ge_k": level["result"],
                                "enumeration_reachable": expected_reachable,
                                "enumerated_maximum_lead": maximum_lead,
                                "query_log": f"e6_{configuration}_lead_ge_{k:02d}_v6.log",
                            }
                        )
                    trace_path = COUNTEREXAMPLES / f"e6_{configuration}_shortest_trace_to_max_lead_v6.json"
                    write_json(
                        trace_path,
                        {
                            "configuration": configuration,
                            "maximum_information_lead_from_exported_reachable_states": maximum_lead,
                            "trace": trace,
                            "pmax_fairness_violation": violation["result"],
                            "pmax_terminal_fairness_violation": terminal_violation["result"],
                            "finite_model_scope": True,
                        },
                    )
                    row = {
                        "configuration": configuration,
                        "configuration_class": "paper_main",
                        "corrupted_party": corrupted_name,
                        "initial_leader": leader_name,
                        "n_blocks": n_blocks,
                        "message_length_bits": message_length_bits,
                        "lambda_bits": lambda_bits,
                        "pmax_fairness_violation": violation["result"],
                        "pmax_terminal_fairness_violation": terminal_violation["result"],
                        "maximum_information_lead_from_reachable_states": maximum_lead,
                        "shortest_trace_length_to_max_lead": len(trace) - 1,
                        "reachable_states": len(states),
                        "reported_prism_states": violation["states"],
                        "transitions": transition_summary["transitions"],
                        "choices": transition_summary["choices"],
                        "construction_seconds": violation["construction_seconds"],
                        "checking_seconds": violation["checking_seconds"],
                        "all_queries_construction_seconds": sum(float(item["construction_seconds"] or 0.0) for item in query_results),
                        "all_queries_checking_seconds": sum(float(item["checking_seconds"] or 0.0) for item in query_results),
                        "peak_memory_bytes": "not_measured",
                        "prism_log": f"e6_{configuration}_violation_v6.log",
                        "terminal_prism_log": f"e6_{configuration}_terminal_violation_v6.log",
                        "model_file": model.name,
                        "model_file_sha256": sha256_file(model),
                        "states_file": states_path.name,
                        "transitions_file": transitions_path.name,
                        "strategy_evidence_kind": "reconstructed_witness_policy",
                        "strategy_file": policy_path.name,
                        "trace_file": trace_path.name,
                        "shortest_abort_prefix_file": abort_trace_path.name,
                        "actions": "normal_send,delay,abort,replay,out_of_order,selective_open,invalid_open,mismatched_randomness,duplicate_old_open,wrong_context",
                        "corrupt_party_own_record_mac": "valid",
                        "scope": "finite_protocol_mdp_under_authentication_and_ideal_commitment_assumptions",
                    }
                    rows.append(row)
                    if float(violation["result"]) != 0.0 or float(terminal_violation["result"]) != 0.0 or maximum_lead > int(lambda_bits):
                        write_json(COUNTEREXAMPLES / f"e6_UNEXPECTED_{configuration}_fairness_violation_v6.json", row)
                    else:
                        write_json(
                            COUNTEREXAMPLES / f"e6_{configuration}_NO_COUNTEREXAMPLE_WITHIN_BOUND_v6.json",
                            {
                                "status": "NO_COUNTEREXAMPLE_WITHIN_BOUND",
                                "configuration": configuration,
                                "pmax_fairness_violation": violation["result"],
                                "enumerated_maximum_lead": maximum_lead,
                                "lambda_bits": lambda_bits,
                                "reachable_states": len(states),
                            },
                        )
    if len(rows) != 48:
        raise AssertionError(f"expected 48 paper-main MDP configurations, got {len(rows)}")
    return rows, level_rows


def run() -> list[dict[str, object]]:
    raw_mc, dtmc = run_dtmc()
    fairness, lead_levels = run_fairness()
    mc_path = RAW / "e6_dtmc_event_level_by_seed_v6.csv"
    dtmc_path = RAW / "e6_dtmc_analytic_prism_mc_v6.csv"
    fairness_path = RAW / "e6_protocol_fairness_mdp_v6.csv"
    levels_path = RAW / "e6_mdp_all_information_lead_levels_v6.csv"
    write_csv(mc_path, raw_mc)
    write_csv(dtmc_path, dtmc)
    write_csv(fairness_path, fairness)
    write_csv(levels_path, lead_levels)
    representative = next(
        row for row in dtmc
        if row["L"] == 32 and math.isclose(float(row["q0"]), 0.03) and math.isclose(float(row["qth"]), 0.10)
    )
    summaries = [
        summary_row("E6", "dtmc_L32_q003_q1020_t010", "honest_false_abort_probability", representative["honest_false_abort_analytic"], ci_low=1-float(representative["honest_acceptance_ci_high"]), ci_high=1-float(representative["honest_acceptance_ci_low"]), unit="probability", raw_hash=sha256_file(dtmc_path)),
        summary_row("E6", "dtmc_L32_q003_q1020_t010", "attack_miss_probability", representative["attack_miss_analytic"], ci_low=representative["attack_miss_ci_low"], ci_high=representative["attack_miss_ci_high"], unit="probability", raw_hash=sha256_file(dtmc_path)),
        summary_row("E6", "dtmc_all_registered", "maximum_prism_analytic_error", max(float(row["analytic_prism_max_error"]) for row in dtmc), unit="probability", raw_hash=sha256_file(dtmc_path)),
        summary_row("E6", "dtmc_all_registered", "maximum_mc_analytic_error", max(float(row["monte_carlo_analytic_max_error"]) for row in dtmc), unit="probability", raw_hash=sha256_file(dtmc_path), notes="Every trial materializes L independent error events."),
        summary_row("E6", "fair_all_registered", "pmax_fairness_violation", max(float(row["pmax_fairness_violation"]) for row in fairness), unit="probability", raw_hash=sha256_file(fairness_path), notes="Finite protocol MDP only; corrupt parties authenticate their own malicious records."),
        summary_row("E6", "fair_all_registered", "maximum_information_lead", max(int(row["maximum_information_lead_from_reachable_states"]) for row in fairness), unit="bits", raw_hash=sha256_file(fairness_path), notes="Computed solely from all PRISM-exported reachable states."),
        summary_row("E6", "fair_all_registered", "model_configurations", len(fairness), unit="configurations", raw_hash=sha256_file(fairness_path)),
        summary_row("E6", "fair_all_registered", "all_k_level_queries", len(lead_levels), unit="queries", raw_hash=sha256_file(levels_path), notes="Each finite MDP configuration queries every k from 0 through message_length_bits."),
    ]
    write_csv(PROCESSED / "e6_summary_v6.csv", summaries)
    return summaries


if __name__ == "__main__":
    run()

