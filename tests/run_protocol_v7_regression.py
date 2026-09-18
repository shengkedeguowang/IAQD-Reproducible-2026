"""Run the bounded Step-05 regression and materialize auditable V7 evidence.

This is intentionally a unit/semantic regression, not a performance or Monte
Carlo experiment.  It writes only to results/derived/step05_protocol_v7.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import io
import json
import sys
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TESTS = ROOT / "tests"
OUTPUT = ROOT / "results" / "derived" / "step05_protocol_v7"
sys.path[:0] = [str(SRC), str(TESTS)]

from protocol_v7 import BASIC, ID_A, ID_B, RELEASE, IAQDSession, ProtocolViolation  # noqa: E402
from protocol_v7_scenarios import run_ablation_catalog, run_attack_catalog, short_tag_exhaustive_check  # noqa: E402
from test_protocol_v7 import finish_release, release_ready_with_leader  # noqa: E402


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path.name}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_events(path: Path, named_logs: list[tuple[str, list[dict[str, Any]]]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for scenario, events in named_logs:
            for event in events:
                handle.write(json.dumps({"scenario": scenario, **event}, ensure_ascii=False, sort_keys=True) + "\n")
                count += 1
    return count


def _behavior_scenarios() -> list[tuple[str, IAQDSession]]:
    scenarios: list[tuple[str, IAQDSession]] = []

    missing = IAQDSession(mode=BASIC, message_a="00", message_b="11")
    missing.send_quantum()
    missing.local_timeout(ID_A, "QRECV_ACK")
    scenarios.append(("qrecv_missing_timeout", missing))

    detection = IAQDSession(mode=BASIC, message_a="00", message_b="11")
    detection.send_quantum()
    ack = detection.send_qrecv_ack()
    detection.deliver(ack, ID_A)
    decoy = detection.send_decoy_info()
    detection.deliver(decoy, ID_B)
    detection.bob_detection(errors=1, trials=2)
    detection.local_timeout(ID_A, "DIALOGUE_B_QPASS")
    scenarios.append(("bob_detection_failure_alice_later_timeout", detection))

    for honest in (ID_A, ID_B):
        session = release_ready_with_leader(honest)
        honest_open = session.send_open(honest, 1)
        corrupt = ID_B if honest == ID_A else ID_A
        session.local_abort(corrupt)
        session.local_timeout(honest, "NEXT_OPEN")
        try:
            session.send_open(honest, 2)
        except ProtocolViolation:
            pass
        # The undelivered record is still a true send and therefore remains in S.
        if hashlib.sha256(honest_open).hexdigest() not in {hashlib.sha256(raw).hexdigest() for raw in session.wire_records}:
            raise AssertionError("honest opening send was not recorded")
        scenarios.append((f"release_stop_honest_{honest.lower()}_leader", session))

    mismatch = IAQDSession(mode=RELEASE, message_a="001011", message_b="110100", lambda_bits=4)
    if not mismatch.run_prelude():
        raise AssertionError("mismatch prelude failed")
    old_mask = mismatch.bob.private_local["own_mask"]
    mismatch.bob.private_local["own_mask"] = ("1" if old_mask[0] == "0" else "0") + old_mask[1:]
    mismatch._event(
        "test_driver",
        "AUDIT_LOCAL_DEVIATION",
        result="Bob changed the private opening mask after forming C_B",
        secret_values_logged=False,
    )
    finish_release(mismatch)
    scenarios.append(("mismatched_mask_not_message_fairness", mismatch))
    return scenarios


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    test_stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(TESTS), pattern="test_protocol_v7.py")
    unit_result = unittest.TextTestRunner(stream=test_stream, verbosity=2).run(suite)

    attacks = run_attack_catalog()
    ablations = run_ablation_catalog(attacks)
    short_tag = short_tag_exhaustive_check()
    basic = IAQDSession(mode=BASIC, message_a="0011", message_b="1100").run_honest()
    release = IAQDSession(mode=RELEASE, message_a="001011", message_b="110100", lambda_bits=4).run_honest()
    behaviors = _behavior_scenarios()

    attack_rows = [
        {
            "name": item.name,
            "legacy_source": item.legacy_source,
            "category": item.category,
            "accepted": item.accepted,
            "reason": item.reason,
            "regression_pass": item.passed,
            "interpretation": item.interpretation,
            "validation_trace": json.dumps(item.validation_trace, ensure_ascii=False, separators=(",", ":")),
        }
        for item in attacks
    ]
    ablation_rows = [dataclasses.asdict(item) for item in ablations]
    _write_csv(OUTPUT / "attack_migration_results_v7.csv", attack_rows)
    _write_csv(OUTPUT / "ablation_migration_results_v7.csv", ablation_rows)

    attack_event_count = _write_events(
        OUTPUT / "attack_event_log_v7.jsonl",
        [(item.name, item.event_log) for item in attacks],
    )
    honest_event_count = _write_events(
        OUTPUT / "honest_event_log_v7.jsonl",
        [("honest_basic", basic.event_log), ("honest_release_nondivisible", release.event_log)],
    )
    behavior_event_count = _write_events(
        OUTPUT / "boundary_event_log_v7.jsonl",
        [(name, session.event_log) for name, session in behaviors],
    )
    short_tag_for_file = {key: value for key, value in short_tag.items() if key != "event_log"}
    _write_json(OUTPUT / "short_tag_exhaustive_v7.json", short_tag_for_file)

    summary = {
        "scope": "bounded semantic/unit regression; no performance experiment and no cryptographic proof",
        "spec_id": "IAQD-REL-CAND-1",
        "executor_version": "IAQD-EXEC-V7",
        "unittest": {
            "tests_run": unit_result.testsRun,
            "failures": len(unit_result.failures),
            "errors": len(unit_result.errors),
            "skipped": len(unit_result.skipped),
            "successful": unit_result.wasSuccessful(),
            "text": test_stream.getvalue(),
        },
        "honest_basic": {
            "success": basic.success,
            "terminal_state": basic.terminal_state,
            "metrics": basic.metrics,
        },
        "honest_release": {
            "success": release.success,
            "terminal_state": release.terminal_state,
            "metrics": release.metrics,
            "alice_max_gap": release.alice_state["max_gap"],
            "bob_max_gap": release.bob_state["max_gap"],
        },
        "legacy_attack_migration": {
            "cases": len(attacks),
            "regression_passes": sum(item.passed for item in attacks),
            "accepted": sum(item.accepted for item in attacks),
        },
        "legacy_ablation_mapping": {
            "cases": len(ablations),
            "statuses": {item.legacy_name: item.migration_status for item in ablations},
        },
        "short_tag": short_tag_for_file,
        "event_rows": {
            "honest": honest_event_count,
            "attacks": attack_event_count,
            "boundaries": behavior_event_count,
        },
        "boundary_behaviors": {
            name: {
                "alice": {
                    "terminal_reason": session.alice.terminal_reason,
                    "S": session.alice.S,
                    "V": session.alice.V,
                    "max_gap": session.alice.max_gap,
                    "recovered_peer_message_present": session.alice.recovered_peer_message is not None,
                },
                "bob": {
                    "terminal_reason": session.bob.terminal_reason,
                    "S": session.bob.S,
                    "V": session.bob.V,
                    "max_gap": session.bob.max_gap,
                    "recovered_peer_message_present": session.bob.recovered_peer_message is not None,
                },
            }
            for name, session in behaviors
        },
        "unexecuted": [
            "V6 pytest suite (pytest/Qiskit/NumPy dependencies unavailable in current bundled Python)",
            "complete performance experiment",
            "PRISM model checking",
            "noise-grid and Monte Carlo experiments",
        ],
    }
    _write_json(OUTPUT / "regression_summary_v7.json", summary)
    print(test_stream.getvalue(), end="")
    print(json.dumps({key: summary[key] for key in ("scope", "unittest", "legacy_attack_migration", "event_rows")}, ensure_ascii=False, indent=2))
    return 0 if unit_result.wasSuccessful() and all(item.passed for item in attacks) and basic.success and release.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
