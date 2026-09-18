#!/usr/bin/env python3
"""Run and audit the IAQD-MODEL-V7 finite PRISM grid.

The runner treats PRISM output as authoritative but does not trust the wrapper's
process exit code alone: any ``Error:`` line or missing result is a failed run.
Reachable-state maxima are calculated from PRISM's explicit state export, not
from cumulative rewards.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "prism" / "iaqd_release_protocol_v7.nm"
PROPS = ROOT / "prism" / "iaqd_release_protocol_v7.props"
NEG_PROPS = ROOT / "prism" / "iaqd_release_negative_v7.props"
PROPERTY_NAMES = [
    "release_violation_pmax",
    "release_reached_pmax",
    "completed_pmax",
    "honest_complete_pmax",
    "rejected_pmax",
    "aborted_pmax",
    "waiting_pmax",
    "terminated_pmin",
    "unexpected_terminal_pmax",
    "retransmit_pmax",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def command_text(parts: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(parts)
    return shlex.join(parts)


def invoke(parts: list[str], log_path: Path) -> tuple[int, str]:
    completed = subprocess.run(
        parts,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    output = completed.stdout
    log_path.write_text(
        "$ " + command_text(parts) + "\n\n" + output,
        encoding="utf-8",
        newline="\n",
    )
    return completed.returncode, output


def parse_prism(output: str, expected_results: int) -> dict[str, object]:
    errors = re.findall(r"(?m)^Error:\s*(.+)$", output)
    results = re.findall(r"(?m)^Result:\s+([^\s(]+)(?:\s+\(exact\))?", output)
    states = re.search(r"(?m)^States:\s+(\d+)\s+\(1 initial\)", output)
    transitions = re.search(r"(?m)^Transitions:\s+(\d+)", output)
    choices = re.search(r"(?m)^Choices:\s+(\d+)", output)
    ok = not errors and len(results) == expected_results and states and transitions and choices
    return {
        "ok": bool(ok),
        "errors": errors,
        "results": results,
        "states": int(states.group(1)) if states else None,
        "transitions": int(transitions.group(1)) if transitions else None,
        "choices": int(choices.group(1)) if choices else None,
    }


def read_states(path: Path) -> tuple[list[str], dict[int, dict[str, object]]]:
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip()
        if not (header.startswith("(") and header.endswith(")")):
            raise ValueError(f"bad state header in {path}: {header!r}")
        names = header[1:-1].split(",")
        states: dict[int, dict[str, object]] = {}
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            state_id_text, values_text = raw.split(":", 1)
            values = values_text.strip()[1:-1].split(",")
            if len(values) != len(names):
                raise ValueError(f"bad state arity in {path}: {raw!r}")
            parsed: list[object] = []
            for value in values:
                if value == "true":
                    parsed.append(True)
                elif value == "false":
                    parsed.append(False)
                else:
                    parsed.append(int(value))
            states[int(state_id_text)] = dict(zip(names, parsed))
    return names, states


def state_maxima(path: Path) -> tuple[int, int, int, dict[int, dict[str, object]]]:
    _, states = read_states(path)
    current = max(int(s["sent_honest"]) - int(s["verified_peer"]) for s in states.values())
    historical = max(int(s["max_gap"]) for s in states.values())
    violations = sum(
        int(s["sent_honest"]) > int(s["verified_peer"]) + CURRENT_LAMBDA
        for s in states.values()
    )
    return current, historical, violations, states


def read_transitions(path: Path) -> dict[int, list[tuple[int, str]]]:
    graph: dict[int, list[tuple[int, str]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        next(handle)
        for raw in handle:
            parts = raw.strip().split(maxsplit=4)
            if len(parts) < 5:
                continue
            source, _choice, target, _probability, action = parts
            graph.setdefault(int(source), []).append((int(target), action))
    return graph


def shortest_trace(
    states: dict[int, dict[str, object]],
    transition_path: Path,
    predicate,
) -> list[dict[str, object]]:
    graph = read_transitions(transition_path)
    queue: deque[int] = deque([0])
    previous: dict[int, tuple[int, str] | None] = {0: None}
    target: int | None = 0 if predicate(states[0]) else None
    while queue and target is None:
        source = queue.popleft()
        for next_state, action in graph.get(source, []):
            if next_state in previous:
                continue
            previous[next_state] = (source, action)
            if predicate(states[next_state]):
                target = next_state
                break
            queue.append(next_state)
    if target is None:
        return []
    edges: list[tuple[int, str, int]] = []
    cursor = target
    while previous[cursor] is not None:
        source, action = previous[cursor]  # type: ignore[misc]
        edges.append((source, action, cursor))
        cursor = source
    edges.reverse()
    fields = [
        "phase",
        "outcome",
        "waiting",
        "sent_blocks_h",
        "verified_blocks_peer",
        "sent_honest",
        "verified_peer",
        "max_gap",
        "last_actor",
        "last_direction",
        "last_type",
        "last_seq",
        "last_block",
        "last_event",
        "reject_reason",
    ]
    rows: list[dict[str, object]] = []
    for step, (source, action, destination) in enumerate(edges, start=1):
        row: dict[str, object] = {
            "step": step,
            "from_state": source,
            "action": action,
            "to_state": destination,
        }
        row.update({field: states[destination][field] for field in fields})
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def exact_is(value: str, expected: str) -> bool:
    try:
        return abs(float(value) - float(expected)) <= 1e-15
    except ValueError:
        return value == expected


def run_configuration(
    prism: Path,
    raw_dir: Path,
    m: int,
    lam: int,
    honest: int,
    first: int,
    variant: int,
    negative: bool,
    export_transitions: bool,
    request_strategy: bool,
    use_exact: bool,
) -> tuple[dict[str, object], dict[int, dict[str, object]], Path | None]:
    global CURRENT_LAMBDA
    CURRENT_LAMBDA = lam
    kind = "negative" if negative else "grid"
    config_id = f"{kind}_m{m}_l{lam}_h{honest}_f{first}_v{variant}"
    logs = raw_dir / "logs"
    states_dir = raw_dir / "states"
    traces_dir = raw_dir / "traces"
    for directory in (logs, states_dir, traces_dir):
        directory.mkdir(parents=True, exist_ok=True)
    state_path = states_dir / f"{config_id}.sta"
    trans_path = traces_dir / f"{config_id}.tra" if export_transitions else None
    strategy_path = traces_dir / f"{config_id}.strategy.txt" if request_strategy else None
    props = NEG_PROPS if negative else PROPS
    expected = 1 if negative else len(PROPERTY_NAMES)
    constants = f"M={m},LAMBDA={lam},HONEST={honest},FIRST={first},VARIANT={variant}"
    cmd = [
        str(prism),
        str(MODEL),
        str(props),
        "-const",
        constants,
        "-explicit",
        "-nofixdl",
        "-exportstates",
        str(state_path),
    ]
    if use_exact:
        cmd.insert(-3, "-exact")
    else:
        cmd[-3:-3] = ["-epsilon", "1e-12"]
    if trans_path is not None:
        cmd.extend(["-exporttrans", str(trans_path)])
    if strategy_path is not None:
        cmd.extend(["-exportstrat", str(strategy_path) + ":type=actions,reach=true,states=true"])
    returncode, output = invoke(cmd, logs / f"{config_id}.log")
    parsed = parse_prism(output, expected)
    if not parsed["ok"] or not state_path.exists():
        raise RuntimeError(
            f"PRISM failure for {config_id}: rc={returncode}, "
            f"errors={parsed['errors']}, results={len(parsed['results'])}/{expected}"
        )
    current_max, historical_max, violating_state_count, states = state_maxima(state_path)
    results = list(parsed["results"])
    row: dict[str, object] = {
        "config_id": config_id,
        "m_bits": m,
        "n_qubit_symbols": m // 2,
        "lambda_bits": lam,
        "chunks": (m + lam - 1) // lam,
        "last_chunk_bits": m - (((m + lam - 1) // lam) - 1) * lam,
        "honest": "Alice" if honest == 0 else "Bob",
        "first": "Alice" if first == 0 else "Bob",
        "variant": variant,
        "engine": "explicit",
        "precision": (
            "exact rational (-exact); epsilon N/A"
            if use_exact
            else "double explicit; termination epsilon=1e-12"
        ),
        "process_exit_code": returncode,
        "output_error_lines": len(parsed["errors"]),
        "states": parsed["states"],
        "transitions": parsed["transitions"],
        "choices": parsed["choices"],
        "reachable_current_gap_max": current_max,
        "reachable_historical_gap_max": historical_max,
        "violating_reachable_states": violating_state_count,
        "state_export": state_path.relative_to(ROOT).as_posix(),
        "transition_export": trans_path.relative_to(ROOT).as_posix() if trans_path else "not requested",
        "strategy_export": (
            strategy_path.relative_to(ROOT).as_posix()
            if strategy_path and strategy_path.exists()
            else "requested but PRISM emitted no file" if strategy_path else "not requested"
        ),
        "log": (logs / f"{config_id}.log").relative_to(ROOT).as_posix(),
    }
    if negative:
        row[PROPERTY_NAMES[0]] = results[0]
    else:
        row.update(dict(zip(PROPERTY_NAMES, results)))
    return row, states, trans_path


def validate_grid(row: dict[str, object]) -> list[str]:
    failures: list[str] = []
    expected = {
        "release_violation_pmax": "0",
        "release_reached_pmax": "1",
        "completed_pmax": "1",
        "honest_complete_pmax": "1",
        "rejected_pmax": "1",
        "aborted_pmax": "1",
        "waiting_pmax": "1",
        "terminated_pmin": "0",
        "unexpected_terminal_pmax": "0",
        "retransmit_pmax": "1",
    }
    for key, wanted in expected.items():
        if not exact_is(str(row[key]), wanted):
            failures.append(f"{row['config_id']} {key}={row[key]} expected {wanted}")
    if int(row["reachable_current_gap_max"]) > int(row["lambda_bits"]):
        failures.append(f"{row['config_id']} current gap exceeds lambda")
    if int(row["reachable_historical_gap_max"]) > int(row["lambda_bits"]):
        failures.append(f"{row['config_id']} historical gap exceeds lambda")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prism", type=Path, default=Path(r"D:\prism-wrapper\prism.bat"))
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    prism = args.prism.resolve()
    if not prism.is_file():
        parser.error(f"PRISM launcher not found: {prism}")
    raw_dir = ROOT / "results" / "raw" / "prism_release_v7" / args.run_id
    derived_dir = ROOT / "results" / "derived" / "prism_release_v7" / args.run_id
    if raw_dir.exists() or derived_dir.exists():
        parser.error("run-id already exists; refusing to overwrite a previous run")
    raw_dir.mkdir(parents=True)
    derived_dir.mkdir(parents=True)

    started = datetime.now(timezone.utc).isoformat()
    grid_rows: list[dict[str, object]] = []
    failures: list[str] = []
    representative: tuple[dict[int, dict[str, object]], Path] | None = None
    configurations = [
        (m, lam, honest, first)
        for m in (8, 16, 32)
        for lam in (1, 2, 4, 8)
        for honest in (0, 1)
        for first in (0, 1)
    ] + [(6, 4, honest, first) for honest in (0, 1) for first in (0, 1)]
    for index, (m, lam, honest, first) in enumerate(configurations, start=1):
        print(f"GRID {index:02d}/{len(configurations)} m={m} lambda={lam} honest={honest} first={first}", flush=True)
        export_transitions = (m, lam, honest, first) == (6, 4, 0, 0)
        row, states, trans_path = run_configuration(
            prism, raw_dir, m, lam, honest, first, 0, False,
            export_transitions, False, False
        )
        grid_rows.append(row)
        failures.extend(validate_grid(row))
        if export_transitions and trans_path is not None:
            representative = (states, trans_path)

    negative_rows: list[dict[str, object]] = []
    negative_payloads: dict[int, tuple[dict[int, dict[str, object]], Path]] = {}
    for variant in (1, 2):
        print(f"NEGATIVE variant={variant}", flush=True)
        row, states, trans_path = run_configuration(
            prism, raw_dir, 8, 2, 0, 0, variant, True,
            True, variant == 2, True
        )
        negative_rows.append(row)
        if trans_path is None:
            raise RuntimeError("negative transition export was not requested")
        negative_payloads[variant] = (states, trans_path)
    if not exact_is(str(negative_rows[0]["release_violation_pmax"]), "0"):
        failures.append("guard-only negative control unexpectedly reached a violation")
    if int(negative_rows[0]["reachable_current_gap_max"]) > 2:
        failures.append("guard-only negative control exceeded lambda")
    if not exact_is(str(negative_rows[1]["release_violation_pmax"]), "1"):
        failures.append("deliberately relaxed negative control failed to reach a violation")
    if int(negative_rows[1]["reachable_current_gap_max"]) <= 2:
        failures.append("deliberately relaxed negative control did not exceed lambda")

    write_csv(derived_dir / "grid_results.csv", grid_rows)
    write_csv(derived_dir / "negative_controls.csv", negative_rows)

    if representative is None:
        raise RuntimeError("representative reliable-delivery graph was not exported")
    honest_states, honest_transitions = representative
    honest_trace = shortest_trace(
        honest_states,
        honest_transitions,
        lambda state: int(state["outcome"]) == 1,
    )
    write_csv(derived_dir / "honest_completion_trace_m6_l4_A_firstA.csv", honest_trace)
    if not honest_trace:
        failures.append("no concrete honest-completion trace found")

    negative_states, negative_transitions = negative_payloads[2]
    violation_trace = shortest_trace(
        negative_states,
        negative_transitions,
        lambda state: int(state["sent_honest"]) > int(state["verified_peer"]) + 2,
    )
    write_csv(derived_dir / "violation_trace_variant2_m8_l2_A_firstA.csv", violation_trace)
    if not violation_trace:
        failures.append("no concrete variant-2 violation trace found")

    summary = {
        "run_id": args.run_id,
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL.relative_to(ROOT).as_posix(),
        "model_sha256": sha256(MODEL),
        "properties": PROPS.relative_to(ROOT).as_posix(),
        "properties_sha256": sha256(PROPS),
        "negative_properties": NEG_PROPS.relative_to(ROOT).as_posix(),
        "negative_properties_sha256": sha256(NEG_PROPS),
        "prism_launcher": str(prism),
        "grid_configurations": len(grid_rows),
        "negative_configurations": len(negative_rows),
        "grid_max_states": max(int(row["states"]) for row in grid_rows),
        "grid_max_transitions": max(int(row["transitions"]) for row in grid_rows),
        "grid_max_reachable_gap": max(int(row["reachable_current_gap_max"]) for row in grid_rows),
        "grid_max_historical_gap": max(int(row["reachable_historical_gap_max"]) for row in grid_rows),
        "normal_violation_queries_nonzero": sum(
            not exact_is(str(row["release_violation_pmax"]), "0") for row in grid_rows
        ),
        "unexpected_terminal_queries_nonzero": sum(
            not exact_is(str(row["unexpected_terminal_pmax"]), "0") for row in grid_rows
        ),
        "negative_guard_only": negative_rows[0],
        "negative_relaxed": negative_rows[1],
        "honest_completion_trace_steps": len(honest_trace),
        "violation_trace_steps": len(violation_trace),
        "validation_failures": failures,
    }
    (derived_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 1 if failures else 0


CURRENT_LAMBDA = 0

if __name__ == "__main__":
    sys.exit(main())
