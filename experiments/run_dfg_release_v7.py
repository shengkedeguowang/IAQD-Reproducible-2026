"""D/F/G: V7 authentication regression, PRISM reuse audit, resources and timing."""

from __future__ import annotations

import dataclasses
import gc
import itertools
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests"), str(ROOT / "experiments")]

from protocol_v7 import BASIC, RELEASE, IAQDSession, ce1_encode, parse_record  # noqa: E402
from protocol_v7_scenarios import run_ablation_catalog, run_attack_catalog, short_tag_exhaustive_check  # noqa: E402
from release_v7_common import (  # noqa: E402
    AUDIT,
    DERIVED,
    LOGS,
    RAW,
    ROOT,
    SEEDS,
    deterministic_protocol_secrets,
    relative,
    sha256_file,
    verify_sha256_manifest,
    write_command_log,
    write_csv,
    write_json,
)


def latest_local_state(event_log: list[dict[str, Any]], actor: str) -> dict[str, Any]:
    for event in reversed(event_log):
        if event.get("actor") == actor and isinstance(event.get("state_after"), dict):
            return dict(event["state_after"])
    return {}


def run_d_authentication() -> dict[str, Any]:
    command = [
        str(ROOT / ".experiment_env_v7" / "Scripts" / "python.exe"),
        "-m",
        "pytest",
        "-q",
        "tests/test_protocol_v7.py",
        "gamma_closure/tests/test_gamma_interface_closure.py",
    ]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    write_command_log(LOGS / "d_pytest_v7_and_gamma.log", command, completed.returncode, completed.stdout, completed.stderr)

    with deterministic_protocol_secrets(SEEDS[0]):
        attacks = run_attack_catalog()
    attack_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    unexpected_accepts: list[dict[str, Any]] = []
    for outcome in attacks:
        alice = latest_local_state(outcome.event_log, "Alice")
        bob = latest_local_state(outcome.event_log, "Bob")
        row = {
            "name": outcome.name,
            "legacy_source": outcome.legacy_source,
            "category": outcome.category,
            "accepted": outcome.accepted,
            "rejected": not outcome.accepted,
            "reason": outcome.reason,
            "regression_pass": outcome.passed,
            "interpretation": outcome.interpretation,
            "validation_trace": json.dumps(outcome.validation_trace, ensure_ascii=False, separators=(",", ":")),
            "input_definition_source": "tests/protocol_v7_scenarios.py",
            "alice_last_local_state": json.dumps(alice, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            "bob_last_local_state": json.dumps(bob, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        }
        attack_rows.append(row)
        if outcome.accepted:
            unexpected_accepts.append(row)
        for event in outcome.event_log:
            event_rows.append({"scenario": outcome.name, **event})
    attack_path = RAW / "d_auth_attack_cases_release_v7.csv"
    event_path = RAW / "d_auth_attack_event_log_release_v7.jsonl"
    write_csv(attack_path, attack_rows)
    with event_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in event_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    write_json(RAW / "d_auth_unexpected_accepts_release_v7.json", unexpected_accepts)

    ablations = run_ablation_catalog(attacks)
    ablation_rows: list[dict[str, Any]] = []
    supporting_executed = 0
    for item in ablations:
        if item.migration_status in {"MAPPED_TO_REQUIRED_CHECK", "MAPPED_TO_SEND_GATE"}:
            execution_kind = "supporting_required-check_case_executed; no weakened implementation"
            supporting_executed += 1
        else:
            execution_kind = "specification_mapping_only; no weakened implementation exists"
        ablation_rows.append({**dataclasses.asdict(item), "execution_kind": execution_kind})
    ablation_path = RAW / "d_auth_ablation_mapping_release_v7.csv"
    write_csv(ablation_path, ablation_rows)

    with deterministic_protocol_secrets(SEEDS[1]):
        short_tag = short_tag_exhaustive_check()
    short_path = RAW / "d_auth_short_tag_exhaustive_release_v7.json"
    write_json(short_path, short_tag)

    honest_rows: list[dict[str, Any]] = []
    honest_event_path = RAW / "d_auth_honest_event_log_release_v7.jsonl"
    with honest_event_path.open("w", encoding="utf-8", newline="\n") as handle:
        for mode, message_bits, lam, seed in ((BASIC, 8, None, SEEDS[2]), (RELEASE, 10, 3, SEEDS[3])):
            message_a = ("00110101" if message_bits == 8 else "0011010110")
            message_b = ("11001010" if message_bits == 8 else "1100101001")
            with deterministic_protocol_secrets(seed):
                result = IAQDSession(mode=mode, message_a=message_a, message_b=message_b, lambda_bits=lam).run_honest()
            honest_rows.append(
                {
                    "mode": mode,
                    "message_bits_per_party": message_bits,
                    "lambda_bits": lam if lam is not None else "",
                    "success": result.success,
                    "terminal_state": result.terminal_state,
                    "records": result.metrics["post_ke_record_count"],
                    "mac_records": result.metrics["mac_record_count"],
                    "prf_confirm_records": result.metrics["prf_confirm_count"],
                    "alice_state": json.dumps(result.alice_state, ensure_ascii=False, sort_keys=True),
                    "bob_state": json.dumps(result.bob_state, ensure_ascii=False, sort_keys=True),
                }
            )
            for event in result.event_log:
                handle.write(json.dumps({"scenario": f"honest_{mode}", **event}, ensure_ascii=False, sort_keys=True) + "\n")
    honest_path = RAW / "d_auth_honest_paths_release_v7.csv"
    write_csv(honest_path, honest_rows)

    return {
        "pytest_exit_code": completed.returncode,
        "pytest_log": relative(LOGS / "d_pytest_v7_and_gamma.log"),
        "attack_cases": len(attacks),
        "attack_regression_passes": sum(item.passed for item in attacks),
        "unexpected_accepts": len(unexpected_accepts),
        "attack_event_rows": len(event_rows),
        "ablation_catalog_rows": len(ablations),
        "ablation_supporting_required_checks_executed": supporting_executed,
        "weakened_protocol_variants_executed": 0,
        "short_tag_candidates": short_tag["candidate_count"],
        "short_tag_reached_prf": short_tag["actual_validator_calls_reaching_prf"],
        "short_tag_accepted": short_tag["accepted_count"],
        "honest_paths": len(honest_rows),
        "honest_successes": sum(bool(row["success"]) for row in honest_rows),
        "hashes": {
            "attacks": sha256_file(attack_path),
            "events": sha256_file(event_path),
            "ablations": sha256_file(ablation_path),
            "short_tag": sha256_file(short_path),
            "honest": sha256_file(honest_path),
        },
        "boundary": "finite semantic regression; ablation mappings are not nine selectable weakened protocol implementations",
    }


def run_f_release_model_reuse() -> dict[str, Any]:
    verification = verify_sha256_manifest(ROOT / "reports" / "PRISM_ALIGNMENT_OUTPUTS.sha256")
    audit_path = DERIVED / "f_release_prism_manifest_audit_release_v7.csv"
    write_csv(audit_path, verification["rows"])
    source = ROOT / "results" / "derived" / "prism_release_v7" / "20260905T200413_CST" / "summary.json"
    summary = json.loads(source.read_text(encoding="utf-8"))
    current_model = sha256_file(ROOT / "prism" / "iaqd_release_protocol_v7.nm")
    current_props = sha256_file(ROOT / "prism" / "iaqd_release_protocol_v7.props")
    current_executor = sha256_file(ROOT / "src" / "protocol_v7.py")
    valid = (
        verification["mismatch"] == 0
        and verification["missing"] == 0
        and verification["match"] == 217
        and current_model == summary["model_sha256"]
        and current_props == summary["properties_sha256"]
        and current_executor == "BB2378E19F4C17A81610DCDFF10F935A9D20A2404092A5A48A01E4F1E4E185BE"
    )
    return {
        "status": "verified_reuse" if valid else "invalidated_pending_rerun",
        "source_run_id": summary["run_id"],
        "manifest_matches": verification["match"],
        "manifest_mismatches": verification["mismatch"],
        "manifest_missing": verification["missing"],
        "normal_configurations": summary["grid_configurations"],
        "negative_configurations": summary["negative_configurations"],
        "normal_nonzero_violation_queries": summary["normal_violation_queries_nonzero"],
        "negative_relaxed_violation_pmax": summary["negative_relaxed"]["release_violation_pmax"],
        "source_summary": relative(source),
        "audit_sha256": sha256_file(audit_path),
        "claim_boundary": "finite model of local S_H/V_H events; not message-knowledge fairness or unbounded cryptographic security",
    }


def compact_decoy_bits(n: int, ell: int) -> int:
    def ceil_log2_comb(total: int, chosen: int) -> int:
        value = math.comb(total, chosen)
        return 0 if value <= 1 else (value - 1).bit_length()

    return 2 * ceil_log2_comb(n + ell, ell) + 4 * ell


def semantic_value_bits(value: Any, key: str = "") -> int:
    """Value-only bits (map keys and CE1 framing excluded), with field-aware binary strings."""

    if isinstance(value, bool):
        return 1
    if isinstance(value, int):
        return max(1, value.bit_length())
    if isinstance(value, bytes):
        return 8 * len(value)
    if isinstance(value, str):
        if key in {"ciphertext", "chunk"} and set(value) <= {"0", "1"}:
            return len(value)
        if key in {"sid", "params_hash", "qctx", "h_ack", "ciphertext_hash", "commit_set_hash", "d_j", "salt", "tag"}:
            try:
                bytes.fromhex(value)
                return 4 * len(value)
            except ValueError:
                pass
        if value in {"X", "Z", "0", "1"}:
            return 1
        return 8 * len(value.encode("utf-8"))
    if isinstance(value, list):
        return sum(semantic_value_bits(item, key) for item in value)
    if isinstance(value, dict):
        return sum(semantic_value_bits(item, child_key) for child_key, item in value.items())
    raise TypeError(type(value))


def operation_counts(result: Any) -> dict[str, int]:
    combined: dict[str, int] = {}
    for state in (result.alice_state, result.bob_state):
        for name, count in state["auth_counters"].items():
            combined[name] = combined.get(name, 0) + int(count)
    return combined


def schedule_rows(mode: str, r: int) -> list[dict[str, Any]]:
    rows = [
        {"layer": 1, "records": "QRECV_ACK", "record_count": 1, "dependency": "quantum delivery/length check"},
        {"layer": 2, "records": "DECOY_INFO", "record_count": 1, "dependency": "verified QRECV_ACK"},
        {"layer": 3, "records": "DIALOGUE_B_QPASS", "record_count": 1, "dependency": "Bob local detection pass"},
        {"layer": 4, "records": "DIALOGUE_A", "record_count": 1, "dependency": "Alice verified DIALOGUE_B_QPASS"},
    ]
    if mode == BASIC:
        rows.append({"layer": 5, "records": "CONFIRM_A|CONFIRM_B", "record_count": 2, "dependency": "each party local full recovery; confirmations independent"})
        return rows
    rows.append({"layer": 5, "records": "COMMIT_SET_A|COMMIT_SET_B", "record_count": 2, "dependency": "both dialogue records available; commitments independent"})
    layer = 6
    for j in range(1, r + 1):
        rows.append({"layer": layer, "records": f"OPEN_leader_{j}", "record_count": 1, "dependency": "previous block complete and local release guard"})
        layer += 1
        rows.append({"layer": layer, "records": f"OPEN_follower_{j}", "record_count": 1, "dependency": f"verified leader OPEN {j}"})
        layer += 1
    rows.append({"layer": layer, "records": "CONFIRM_A|CONFIRM_B", "record_count": 2, "dependency": "each party S=V=2n and local full recovery"})
    return rows


def run_one_resource(mode: str, n: int, ell: int, lam: int | None, seed: int) -> tuple[Any, list[dict[str, Any]]]:
    message_a = "01" * n
    message_b = "10" * n
    with deterministic_protocol_secrets(seed):
        result = IAQDSession(mode=mode, message_a=message_a, message_b=message_b, lambda_bits=lam, ell=ell).run_honest()
    if not result.success:
        raise AssertionError(f"resource path failed: {mode},n={n},ell={ell},lambda={lam}")
    records = [parse_record(raw) for raw in result.wire_records]
    detail = []
    for index, (raw, record) in enumerate(zip(result.wire_records, records), 1):
        detail.append(
            {
                "mode": mode,
                "n_blocks": n,
                "ell": ell,
                "lambda_bits": lam if lam is not None else "",
                "record_index": index,
                "sender": record["hdr"]["sender"],
                "direction": record["hdr"]["direction"],
                "record_type": record["hdr"]["type"],
                "sequence": record["hdr"]["seq"],
                "auth_kind": record["auth"]["kind"],
                "serialized_bytes": len(raw),
                "header_ce1_bytes": len(ce1_encode(record["hdr"])),
                "payload_ce1_bytes": len(ce1_encode(record["payload"])),
                "auth_ce1_bytes": len(ce1_encode(record["auth"])),
                "auth_tag_bits": 4 * len(record["auth"]["tag"]),
                "value_only_semantic_bits": semantic_value_bits(record),
            }
        )
    return result, detail


def run_g_resources_and_performance() -> dict[str, Any]:
    resource_rows: list[dict[str, Any]] = []
    record_rows: list[dict[str, Any]] = []
    round_rows: list[dict[str, Any]] = []
    quantum_rows: list[dict[str, Any]] = []
    configurations: list[tuple[str, int, int, int | None]] = []
    for n, ell in itertools.product((1, 2, 4, 8, 16, 32), (4, 155)):
        configurations.append((BASIC, n, ell, None))
        for lam in (1, 2, 4, 8):
            if lam <= 2 * n:
                configurations.append((RELEASE, n, ell, lam))

    for config_index, (mode, n, ell, lam) in enumerate(configurations):
        result, details = run_one_resource(mode, n, ell, lam, SEEDS[0] + config_index)
        record_rows.extend(details)
        r = int(result.metrics["r"])
        schedule = schedule_rows(mode, r)
        round_rows.extend({"mode": mode, "n_blocks": n, "ell": ell, "lambda_bits": lam if lam is not None else "", "r": r, **row} for row in schedule)
        counters = operation_counts(result)
        d_bits = compact_decoy_bits(n, ell)
        comparable = (
            d_bits + 4 * n + 1536
            if mode == BASIC
            else d_bits + 8 * n + 2560 + 1792 * r
        )
        resource_rows.append(
            {
                "scheme": f"IAQD_{mode}_V7",
                "mode": mode,
                "n_blocks": n,
                "message_bits_per_party": 2 * n,
                "ell": ell,
                "lambda_bits": lam if lam is not None else "",
                "r": r,
                "last_chunk_bits": (2 * n - (r - 1) * int(lam)) if mode == RELEASE else "",
                "KE_QKD_communication_included": False,
                "initialization_records_separate": 2,
                "post_ke_record_count": len(result.wire_records),
                "causal_round_count_post_ke": len(schedule),
                "quantum_transmissions": 1,
                "actual_serialized_post_ke_bytes": sum(len(raw) for raw in result.wire_records),
                "actual_value_only_semantic_bits": sum(semantic_value_bits(parse_record(raw)) for raw in result.wire_records),
                "comparable_payload_auth_binary_bits": comparable,
                "comparable_formula": "D_dec+4n+1536" if mode == BASIC else "D_dec+8n+2560+1792r",
                "D_dec_compact_bits": d_bits,
                "mac_records": result.metrics["mac_record_count"],
                "prf_confirm_records": result.metrics["prf_confirm_count"],
                "mac_generate_calls": sum(value for key, value in counters.items() if key.endswith("mac_generate")),
                "mac_verify_calls": sum(value for key, value in counters.items() if key.endswith("mac_verify")),
                "prf_generate_calls": counters.get("confirm_prf_generate", 0),
                "prf_verify_calls": counters.get("confirm_prf_verify", 0),
                "commitment_generate_operations": 2 * r if mode == RELEASE else 0,
                "commitment_verify_operations": 2 * r if mode == RELEASE else 0,
                "open_records": 2 * r if mode == RELEASE else 0,
                "claim_boundary": "reference CE1 implementation; in-memory network; initialization and KE/QKD excluded from byte total",
            }
        )
    for n, ell in itertools.product((1, 2, 4, 8, 16, 32), (4, 155)):
        quantum_rows.extend(
            {
                "scheme": scheme,
                "n_blocks": n,
                "ell": ell,
                "prepared_qubits": 6 * n + 2 * ell,
                "transmitted_qubits": 2 * n + 2 * ell,
                "stored_qubits": 4 * n,
                "bell_measurement_operations": 3 * n,
                "bell_readout_bits": 6 * n,
                "decoy_readout_bits": 2 * ell,
                "total_measurement_readout_bits": 6 * n + 2 * ell,
                "joint_6n_state_constructed": False,
                "source": "six-particle block protocol count; same quantum core for all three schemes",
            }
            for scheme in ("original_AQD", "IAQD_BASIC_V7", "IAQD_RELEASE_V7")
        )

    # Original AQD has no accepted CE1 implementation.  Keep the comparable
    # symbolic field count and make the serialization gap explicit.
    for n, ell in itertools.product((1, 2, 4, 8, 16, 32), (4, 155)):
        d_bits = compact_decoy_bits(n, ell)
        resource_rows.append(
            {
                "scheme": "original_AQD",
                "mode": "ORIGINAL",
                "n_blocks": n,
                "message_bits_per_party": 2 * n,
                "ell": ell,
                "lambda_bits": "",
                "r": "",
                "last_chunk_bits": "",
                "KE_QKD_communication_included": False,
                "initialization_records_separate": "not_defined_for_original_paper_protocol",
                "post_ke_record_count": 6,
                "causal_round_count_post_ke": 5,
                "quantum_transmissions": 1,
                "actual_serialized_post_ke_bytes": "NA_no_common_serializer",
                "actual_value_only_semantic_bits": "NA_no_common_serializer",
                "comparable_payload_auth_binary_bits": 256 + d_bits + 4 * n + 2 * 256,
                "comparable_formula": "rho(256)+D_dec+4n+2d_H(256)",
                "D_dec_compact_bits": d_bits,
                "mac_records": "NA",
                "prf_confirm_records": 0,
                "mac_generate_calls": "NA",
                "mac_verify_calls": "NA",
                "prf_generate_calls": 0,
                "prf_verify_calls": 0,
                "commitment_generate_operations": 0,
                "commitment_verify_operations": 0,
                "open_records": 0,
                "claim_boundary": "paper symbolic field count; no byte-level comparison because original AQD has no reviewed CE1 serializer",
            }
        )

    resource_path = RAW / "g_resource_accounting_release_v7.csv"
    record_path = RAW / "g_wire_records_release_v7.csv"
    round_path = RAW / "g_causal_round_schedule_release_v7.csv"
    quantum_path = RAW / "g_quantum_resource_counts_release_v7.csv"
    write_csv(resource_path, resource_rows)
    write_csv(record_path, record_rows)
    write_csv(round_path, round_rows)
    write_csv(quantum_path, quantum_rows)

    performance_rows: list[dict[str, Any]] = []
    for config_index, (mode, n, ell, lam) in enumerate(configurations):
        message_a = "01" * n
        message_b = "10" * n
        for warmup in range(5):
            with deterministic_protocol_secrets(SEEDS[warmup % len(SEEDS)] + 10_000_000 + config_index * 100 + warmup):
                warm = IAQDSession(mode=mode, message_a=message_a, message_b=message_b, lambda_bits=lam, ell=ell).run_honest()
            if not warm.success:
                raise AssertionError("warmup failed")
        for run_index in range(30):
            seed = SEEDS[run_index % len(SEEDS)]
            secret_seed = seed + 20_000_000 + config_index * 1000 + run_index
            with deterministic_protocol_secrets(secret_seed):
                started = time.perf_counter_ns()
                result = IAQDSession(mode=mode, message_a=message_a, message_b=message_b, lambda_bits=lam, ell=ell).run_honest()
                elapsed_ns = time.perf_counter_ns() - started
            performance_rows.append(
                {
                    "mode": mode,
                    "n_blocks": n,
                    "ell": ell,
                    "lambda_bits": lam if lam is not None else "",
                    "r": result.metrics["r"],
                    "run_index": run_index,
                    "input_seed": seed,
                    "fixture_secret_seed": secret_seed,
                    "warmups_before_measurement": 5,
                    "elapsed_ns": elapsed_ns,
                    "elapsed_ms": elapsed_ns / 1_000_000,
                    "success": result.success,
                    "post_ke_records": result.metrics["post_ke_record_count"],
                    "wire_bytes": sum(len(raw) for raw in result.wire_records),
                    "timing_scope": "in-memory V7 session incl simulated KE derivation and deterministic quantum fixture; excl external KE/QKD, network, Qiskit, PRISM",
                    "timer": "time.perf_counter_ns",
                    "process": "single_process",
                }
            )
    performance_path = RAW / "g_performance_runs_release_v7.csv"
    write_csv(performance_path, performance_rows)

    aggregate_rows: list[dict[str, Any]] = []
    for mode, n, ell, lam in configurations:
        subset = [
            float(row["elapsed_ms"])
            for row in performance_rows
            if row["mode"] == mode and row["n_blocks"] == n and row["ell"] == ell and row["lambda_bits"] == (lam if lam is not None else "")
        ]
        q = np.quantile(subset, [0.05, 0.25, 0.5, 0.75, 0.95], method="linear")
        aggregate_rows.append(
            {
                "mode": mode,
                "n_blocks": n,
                "ell": ell,
                "lambda_bits": lam if lam is not None else "",
                "r": 0 if mode == BASIC else math.ceil(2 * n / int(lam)),
                "warmup_runs": 5,
                "measured_runs": len(subset),
                "mean_ms": statistics.fmean(subset),
                "median_ms": float(q[2]),
                "sample_sd_ms": statistics.stdev(subset),
                "p05_ms": float(q[0]),
                "p25_ms": float(q[1]),
                "p75_ms": float(q[3]),
                "p95_ms": float(q[4]),
                "min_ms": min(subset),
                "max_ms": max(subset),
                "successful_runs": len(subset),
            }
        )
    aggregate_path = DERIVED / "g_performance_aggregate_release_v7.csv"
    write_csv(aggregate_path, aggregate_rows)
    return {
        "v7_configurations": len(configurations),
        "resource_rows_including_original": len(resource_rows),
        "wire_record_rows": len(record_rows),
        "causal_schedule_rows": len(round_rows),
        "quantum_resource_rows": len(quantum_rows),
        "performance_warmup_runs": len(configurations) * 5,
        "performance_measured_runs": len(performance_rows),
        "performance_failed_runs": sum(not bool(row["success"]) for row in performance_rows),
        "hashes": {
            "resource": sha256_file(resource_path),
            "records": sha256_file(record_path),
            "rounds": sha256_file(round_path),
            "quantum": sha256_file(quantum_path),
            "performance_raw": sha256_file(performance_path),
            "performance_aggregate": sha256_file(aggregate_path),
        },
        "serialization_boundary": "Original AQD lacks the reviewed CE1 serializer, so its byte column is NA rather than an invented comparable number.",
    }


def main() -> int:
    summary = {
        "D_authentication": run_d_authentication(),
        "F_release_MDP": run_f_release_model_reuse(),
        "G_resources_performance": run_g_resources_and_performance(),
    }
    write_json(DERIVED / "dfg_summary_release_v7.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["D_authentication"]["pytest_exit_code"] == 0 and summary["D_authentication"]["unexpected_accepts"] == 0 and summary["F_release_MDP"]["status"] == "verified_reuse" else 1


if __name__ == "__main__":
    raise SystemExit(main())
