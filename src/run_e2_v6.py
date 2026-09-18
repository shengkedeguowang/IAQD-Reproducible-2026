from __future__ import annotations

import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from scipy.stats import binomtest

from common_v6 import (
    PROCESSED,
    RAW,
    clopper_pearson_upper_zero,
    holm_adjust,
    runtime_config,
    seeds,
    sha256_file,
    summary_row,
    wilson_interval,
    write_csv,
)


STRATEGIES = ("random_replacement", "random_basis_imr", "fixed_z_imr", "biased_basis_imr")


def eve_z_probability(strategy: str, configured_bias: float) -> float:
    if strategy in {"random_replacement", "random_basis_imr"}:
        return 0.5
    if strategy == "fixed_z_imr":
        return 1.0
    if strategy == "biased_basis_imr":
        return configured_bias
    raise ValueError(strategy)


def attacked_pass_probability(strategy: str, alice_z: float, configured_bias: float) -> float:
    if strategy == "random_replacement":
        return 0.5
    eve_z = eve_z_probability(strategy, configured_bias)
    basis_match = alice_z * eve_z + (1 - alice_z) * (1 - eve_z)
    return 0.5 + 0.5 * basis_match


def analytic_undetected(strategy: str, length: int, fraction: float, alice_z: float, configured_bias: float) -> float:
    single = (1 - fraction) + fraction * attacked_pass_probability(strategy, alice_z, configured_bias)
    return single**length


def simulate_event_batch(rng: np.random.Generator, trials: int, length: int, fraction: float, strategy: str, alice_z: float, configured_bias: float) -> tuple[int, int, int]:
    shape = (trials, length)
    alice_basis = (rng.random(shape) >= alice_z).astype(np.uint8)  # 0=Z, 1=X
    alice_bit = rng.integers(0, 2, size=shape, dtype=np.uint8)
    attacked = rng.random(shape) < fraction
    eve_z = eve_z_probability(strategy, configured_bias)
    eve_basis = (rng.random(shape) >= eve_z).astype(np.uint8)
    if strategy == "random_replacement":
        eve_result = rng.integers(0, 2, size=shape, dtype=np.uint8)
    else:
        measurement_random = rng.integers(0, 2, size=shape, dtype=np.uint8)
        eve_result = np.where(eve_basis == alice_basis, alice_bit, measurement_random).astype(np.uint8)
    bob_random = rng.integers(0, 2, size=shape, dtype=np.uint8)
    bob_result_if_attacked = np.where(eve_basis == alice_basis, eve_result, bob_random).astype(np.uint8)
    bob_result = np.where(attacked, bob_result_if_attacked, alice_bit).astype(np.uint8)
    particle_pass = bob_result == alice_bit
    trial_undetected = np.all(particle_pass, axis=1)
    return int(np.count_nonzero(trial_undetected)), int(np.count_nonzero(particle_pass & attacked)), int(np.count_nonzero(attacked))


def simulate_seed(seed_value: int, trials: int, batch_trials: int, length: int, fraction: float, strategy: str, alice_z: float, configured_bias: float) -> dict[str, int]:
    rng = np.random.default_rng(seed_value)
    undetected = 0
    attacked_particle_passes = 0
    attacked_particles = 0
    completed = 0
    while completed < trials:
        current = min(batch_trials, trials - completed)
        passed, particle_passes, particle_count = simulate_event_batch(
            rng, current, length, fraction, strategy, alice_z, configured_bias
        )
        undetected += passed
        attacked_particle_passes += particle_passes
        attacked_particles += particle_count
        completed += current
    return {
        "undetected_trials": undetected,
        "attacked_particle_passes": attacked_particle_passes,
        "attacked_particles": attacked_particles,
    }


def qiskit_imr_circuit(prepared_basis: int, prepared_bit: int, eve_basis: int) -> QuantumCircuit:
    circuit = QuantumCircuit(1, 2, name="single_particle_imr")
    if prepared_bit:
        circuit.x(0)
    if prepared_basis:
        circuit.h(0)
    if eve_basis:
        circuit.h(0)
    circuit.measure(0, 0)
    circuit.reset(0)
    with circuit.if_test((circuit.clbits[0], True)):
        circuit.x(0)
    if eve_basis:
        circuit.h(0)
    if prepared_basis:
        circuit.h(0)
    circuit.measure(0, 1)
    return circuit


def qiskit_single_particle_rows(shots: int) -> list[dict[str, object]]:
    simulator = AerSimulator()
    circuits: list[QuantumCircuit] = []
    metadata: list[tuple[int, int, int]] = []
    for prepared_basis in (0, 1):
        for prepared_bit in (0, 1):
            for eve_basis in (0, 1):
                circuits.append(qiskit_imr_circuit(prepared_basis, prepared_bit, eve_basis))
                metadata.append((prepared_basis, prepared_bit, eve_basis))
    result = simulator.run(circuits, shots=shots, seed_simulator=seeds()[0]).result()
    rows: list[dict[str, object]] = []
    for index, (prepared_basis, prepared_bit, eve_basis) in enumerate(metadata):
        counts = result.get_counts(index)
        passes = sum(count for key, count in counts.items() if key.replace(" ", "")[0] == str(prepared_bit))
        rows.append(
            {
                "prepared_basis": "X" if prepared_basis else "Z",
                "prepared_bit": prepared_bit,
                "eve_basis": "X" if eve_basis else "Z",
                "shots": shots,
                "passes": passes,
                "pass_rate": passes / shots,
                "seed_simulator": seeds()[0],
                "circuit_operations": ";".join(instruction.operation.name for instruction in circuits[index].data),
            }
        )
    return rows


def run() -> list[dict[str, object]]:
    config = runtime_config()["e2"]
    trials_per_seed = int(config["trials_per_seed"])
    batch_trials = int(config["event_batch_trials"])
    alice_z = float(config["alice_z_probability"])
    configured_bias = float(config["adversary_z_probability"])
    total_trials = trials_per_seed * len(seeds())
    raw_rows: list[dict[str, object]] = []
    aggregate_rows: list[dict[str, object]] = []
    p_values: list[float] = []
    adjusted_indexes: list[int] = []
    for strategy in STRATEGIES:
        for length in (8, 16, 24, 32, 64):
            for fraction in (0.10, 0.25, 0.50, 0.75, 1.00):
                analytic = analytic_undetected(strategy, length, fraction, alice_z, configured_bias)
                configuration = f"{strategy}_L{length}_f{fraction:g}"
                run_mc = analytic * total_trials >= int(config["rare_expected_events_threshold"])
                if run_mc:
                    total_undetected = 0
                    particle_passes = 0
                    particle_attacks = 0
                    for seed_value in seeds():
                        observed = simulate_seed(
                            seed_value + 1009 * STRATEGIES.index(strategy) + length,
                            trials_per_seed,
                            batch_trials,
                            length,
                            fraction,
                            strategy,
                            alice_z,
                            configured_bias,
                        )
                        total_undetected += observed["undetected_trials"]
                        particle_passes += observed["attacked_particle_passes"]
                        particle_attacks += observed["attacked_particles"]
                        raw_rows.append(
                            {
                                "configuration": configuration,
                                "strategy": strategy,
                                "L": length,
                                "attack_fraction": fraction,
                                "seed": seed_value,
                                "trials": trials_per_seed,
                                "undetected_events": observed["undetected_trials"],
                                "empirical_undetected": observed["undetected_trials"] / trials_per_seed,
                                "attacked_particle_passes": observed["attacked_particle_passes"],
                                "attacked_particles": observed["attacked_particles"],
                                "single_attacked_particle_pass_rate": observed["attacked_particle_passes"] / observed["attacked_particles"] if observed["attacked_particles"] else "",
                                "analytic_undetected": analytic,
                                "mc_executed": True,
                                "simulation_level": "explicit_BB84_particle_events",
                            }
                        )
                    low, high = wilson_interval(total_undetected, total_trials)
                    empirical = total_undetected / total_trials
                    p_value = float(binomtest(total_undetected, total_trials, analytic).pvalue)
                    aggregate_rows.append(
                        {
                            "configuration": configuration,
                            "strategy": strategy,
                            "L": length,
                            "attack_fraction": fraction,
                            "analytic_undetected": analytic,
                            "analytic_detection": 1 - analytic,
                            "empirical_undetected": empirical,
                            "empirical_detection": 1 - empirical,
                            "undetected_ci_low": low,
                            "undetected_ci_high": high,
                            "detection_ci_low": 1 - high,
                            "detection_ci_high": 1 - low,
                            "absolute_error": abs(empirical - analytic),
                            "single_attacked_particle_pass_rate": particle_passes / particle_attacks if particle_attacks else "",
                            "mc_executed": True,
                            "zero_count_upper_95": "",
                            "p_value": p_value,
                            "holm_adjusted_p": "",
                        }
                    )
                    p_values.append(p_value)
                    adjusted_indexes.append(len(aggregate_rows) - 1)
                else:
                    upper = clopper_pearson_upper_zero(total_trials)
                    raw_rows.append(
                        {
                            "configuration": configuration,
                            "strategy": strategy,
                            "L": length,
                            "attack_fraction": fraction,
                            "seed": "",
                            "trials": 0,
                            "analytic_undetected": analytic,
                            "mc_executed": False,
                            "simulation_level": "analytic_rare_event_only",
                            "zero_count_upper_95": upper,
                        }
                    )
                    aggregate_rows.append(
                        {
                            "configuration": configuration,
                            "strategy": strategy,
                            "L": length,
                            "attack_fraction": fraction,
                            "analytic_undetected": analytic,
                            "analytic_detection": 1 - analytic,
                            "empirical_undetected": "",
                            "empirical_detection": "",
                            "undetected_ci_low": "",
                            "undetected_ci_high": "",
                            "detection_ci_low": "",
                            "detection_ci_high": "",
                            "absolute_error": "",
                            "single_attacked_particle_pass_rate": "",
                            "mc_executed": False,
                            "zero_count_upper_95": upper,
                            "p_value": "",
                            "holm_adjusted_p": "",
                        }
                    )
    for index, adjusted in zip(adjusted_indexes, holm_adjust(p_values)):
        aggregate_rows[index]["holm_adjusted_p"] = adjusted

    qiskit_rows = qiskit_single_particle_rows(int(config["qiskit_shots"]))
    raw_path = RAW / "e2_imr_event_level_by_seed_v6.csv"
    aggregate_path = PROCESSED / "e2_imr_event_level_aggregate_v6.csv"
    qiskit_path = RAW / "e2_imr_single_particle_qiskit_v6.csv"
    write_csv(raw_path, raw_rows)
    write_csv(aggregate_path, aggregate_rows)
    write_csv(qiskit_path, qiskit_rows)
    lookup = {str(row["configuration"]): row for row in aggregate_rows}
    maximum_error = max(float(row["absolute_error"]) for row in aggregate_rows if row["absolute_error"] != "")
    single_random = simulate_seed(seeds()[0], 300000, 10000, 1, 1.0, "random_basis_imr", alice_z, configured_bias)
    single_replace = simulate_seed(seeds()[0], 300000, 10000, 1, 1.0, "random_replacement", alice_z, configured_bias)
    random_rate = single_random["undetected_trials"] / 300000
    replace_rate = single_replace["undetected_trials"] / 300000
    if abs(random_rate - 0.75) > 0.01 or abs(replace_rate - 0.5) > 0.01:
        raise AssertionError("single-particle event simulation failed physical sanity check")
    summaries = [
        summary_row("E2", "random_basis_imr_L16_f1", "detection_probability", lookup["random_basis_imr_L16_f1"]["empirical_detection"], ci_low=lookup["random_basis_imr_L16_f1"]["detection_ci_low"], ci_high=lookup["random_basis_imr_L16_f1"]["detection_ci_high"], unit="probability", raw_hash=sha256_file(raw_path), notes="Event-level BB84 particle simulation; analytic formula is a separate reference."),
        summary_row("E2", "random_replacement_L16_f1", "detection_probability", lookup["random_replacement_L16_f1"]["empirical_detection"], ci_low=lookup["random_replacement_L16_f1"]["detection_ci_low"], ci_high=lookup["random_replacement_L16_f1"]["detection_ci_high"], unit="probability", raw_hash=sha256_file(raw_path)),
        summary_row("E2", "all_observable_configurations", "maximum_absolute_error", maximum_error, unit="probability", raw_hash=sha256_file(aggregate_path)),
        summary_row("E2", "single_particle_random_basis_f1", "pass_probability", random_rate, unit="probability", raw_hash=sha256_file(raw_path), notes="Expected 3/4."),
        summary_row("E2", "single_particle_random_replacement_f1", "pass_probability", replace_rate, unit="probability", raw_hash=sha256_file(raw_path), notes="Expected 1/2."),
    ]
    write_csv(PROCESSED / "e2_summary_v6.csv", summaries)
    return summaries


if __name__ == "__main__":
    run()

