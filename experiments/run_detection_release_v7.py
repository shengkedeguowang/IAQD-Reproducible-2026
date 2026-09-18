"""E detection DTMC: analytic, PRISM and event-level sampling on one boundary."""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments")]

from release_v7_common import (  # noqa: E402
    DERIVED,
    LOGS,
    RAW,
    ROOT,
    SEEDS,
    clopper_pearson_upper_zero,
    exact_binomial_cdf,
    run_prism,
    sha256_file,
    wilson_interval,
    write_csv,
    write_json,
)


def simulate_error_counts(base_seed: int, length: int, q: float, trials: int = 100000, batch: int = 5000) -> np.ndarray:
    derived = int(np.random.SeedSequence([base_seed, length, int(round(q * 1_000_000)), 0xE6]).generate_state(1)[0])
    rng = np.random.default_rng(derived)
    counts = np.empty(trials, dtype=np.uint16)
    offset = 0
    while offset < trials:
        current = min(batch, trials - offset)
        # Each cell is a separately generated decoy-error event.  The terminal
        # pass/fail bit is not sampled from the closed-form binomial tail.
        events = rng.random((current, length), dtype=np.float32) < q
        counts[offset : offset + current] = np.sum(events, axis=1, dtype=np.uint16)
        offset += current
    return counts


def prism_number(text: str) -> float:
    approximations = re.findall(r"\((?:~)?([-+0-9.eE]+)\)", text)
    if approximations:
        return float(approximations[-1])
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        return int(numerator.strip()) / int(denominator.strip())
    return float(text.strip())


def main() -> int:
    lengths = (16, 32, 64, 128)
    q0_values = (0.0, 0.01, 0.03, 0.05)
    q1 = 0.20
    thresholds = (0.05, 0.10, 0.15)
    trials_per_seed = 100000
    raw_rows: list[dict[str, object]] = []
    aggregate_rows: list[dict[str, object]] = []
    prism_queries = 0

    for length in lengths:
        cache: dict[tuple[float, int], np.ndarray] = {}
        for q in (*q0_values, q1):
            for seed in SEEDS:
                cache[(q, seed)] = simulate_error_counts(seed, length, q, trials_per_seed, 5000)
        for q0 in q0_values:
            for qth in thresholds:
                threshold_count = math.floor(length * qth)
                config = f"L{length}_q0{q0:g}_q1{q1:g}_qth{qth:g}"
                totals: dict[str, int] = {"honest": 0, "attack": 0}
                for role, q in (("honest", q0), ("attack", q1)):
                    for seed in SEEDS:
                        accepted = int(np.count_nonzero(cache[(q, seed)] <= threshold_count))
                        totals[role] += accepted
                        raw_rows.append(
                            {
                                "configuration": config,
                                "role": role,
                                "seed": seed,
                                "L": length,
                                "q0": q0,
                                "q1": q1,
                                "q": q,
                                "q_threshold": qth,
                                "threshold_count": threshold_count,
                                "accept_rule": "errors<=floor(L*q_threshold)",
                                "executor_equivalent": f"errors*100<={length}*{int(round(qth*100))}",
                                "trials": trials_per_seed,
                                "accepted": accepted,
                                "rejected": trials_per_seed - accepted,
                                "acceptance_rate": accepted / trials_per_seed,
                                "simulation_level": "L_independent_error_events_per_trial",
                            }
                        )

                total_trials = trials_per_seed * len(SEEDS)
                honest_analytic = exact_binomial_cdf(length, q0, threshold_count)
                attack_analytic = exact_binomial_cdf(length, q1, threshold_count)
                honest_prism = run_prism(
                    ROOT / "prism" / "iaqd_detection_v6.pm",
                    ROOT / "prism" / "iaqd_detection_v6.props",
                    constants={"L": length, "T": threshold_count, "q": f"{q0:.12g}"},
                    log_path=LOGS / f"e_detection_{config}_honest.log",
                )
                attack_prism = run_prism(
                    ROOT / "prism" / "iaqd_detection_v6.pm",
                    ROOT / "prism" / "iaqd_detection_v6.props",
                    constants={"L": length, "T": threshold_count, "q": f"{q1:.12g}"},
                    log_path=LOGS / f"e_detection_{config}_attack.log",
                )
                prism_queries += 2
                honest_prism_value = prism_number(str(honest_prism["results"][-1]))
                attack_prism_value = prism_number(str(attack_prism["results"][-1]))
                honest_low, honest_high = wilson_interval(totals["honest"], total_trials)
                attack_low, attack_high = wilson_interval(totals["attack"], total_trials)
                honest_emp = totals["honest"] / total_trials
                attack_emp = totals["attack"] / total_trials
                aggregate_rows.append(
                    {
                        "configuration": config,
                        "L": length,
                        "q0": q0,
                        "q1": q1,
                        "q_threshold": qth,
                        "threshold_count": threshold_count,
                        "accept_rule": "errors<=floor(L*q_threshold)",
                        "seeds": len(SEEDS),
                        "trials_per_seed": trials_per_seed,
                        "total_trials_per_role": total_trials,
                        "honest_accepted": totals["honest"],
                        "honest_acceptance_empirical": honest_emp,
                        "honest_acceptance_wilson_low": honest_low,
                        "honest_acceptance_wilson_high": honest_high,
                        "honest_zero_abort_upper_95": clopper_pearson_upper_zero(total_trials) if totals["honest"] == total_trials else "",
                        "honest_acceptance_analytic": honest_analytic,
                        "honest_acceptance_prism": honest_prism_value,
                        "attack_accepted": totals["attack"],
                        "attack_miss_empirical": attack_emp,
                        "attack_miss_wilson_low": attack_low,
                        "attack_miss_wilson_high": attack_high,
                        "attack_zero_accept_upper_95": clopper_pearson_upper_zero(total_trials) if totals["attack"] == 0 else "",
                        "attack_miss_analytic": attack_analytic,
                        "attack_miss_prism": attack_prism_value,
                        "analytic_prism_max_abs_error": max(abs(honest_analytic - honest_prism_value), abs(attack_analytic - attack_prism_value)),
                        "empirical_analytic_max_abs_error": max(abs(honest_emp - honest_analytic), abs(attack_emp - attack_analytic)),
                        "honest_states": honest_prism["states"],
                        "honest_transitions": honest_prism["transitions"],
                        "attack_states": attack_prism["states"],
                        "attack_transitions": attack_prism["transitions"],
                        "honest_prism_log": honest_prism["log"],
                        "attack_prism_log": attack_prism["log"],
                    }
                )

    raw_path = RAW / "e_detection_event_by_seed_release_v7.csv"
    aggregate_path = DERIVED / "e_detection_analytic_prism_mc_release_v7.csv"
    write_csv(raw_path, raw_rows)
    write_csv(aggregate_path, aggregate_rows)
    summary = {
        "configurations": len(aggregate_rows),
        "raw_rows": len(raw_rows),
        "prism_queries": prism_queries,
        "event_trials_per_role": sum(int(row["total_trials_per_role"]) for row in aggregate_rows),
        "max_analytic_prism_abs_error": max(float(row["analytic_prism_max_abs_error"]) for row in aggregate_rows),
        "max_empirical_analytic_abs_error": max(float(row["empirical_analytic_max_abs_error"]) for row in aggregate_rows),
        "zero_attack_accept_configurations": sum(int(row["attack_accepted"]) == 0 for row in aggregate_rows),
        "raw_sha256": sha256_file(raw_path),
        "aggregate_sha256": sha256_file(aggregate_path),
        "claim_boundary": "finite independent-error DTMC/event model; not a general quantum attack proof",
    }
    write_json(DERIVED / "e_detection_summary_release_v7.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
