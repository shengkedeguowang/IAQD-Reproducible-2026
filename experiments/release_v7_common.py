"""Shared, version-locked helpers for IAQD-EXP-REL-1.

This module writes only into the run-specific raw/derived/log/audit folders.
It never changes the accepted protocol, theory, tests, PRISM models, or V6 data.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import re
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import yaml


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "experiments" / "experiment_manifest_release_v7.yaml"
MANIFEST = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
RUN_ID = str(MANIFEST["run_id"])
RAW = ROOT / str(MANIFEST["paths"]["raw"])
DERIVED = ROOT / str(MANIFEST["paths"]["derived"])
LOGS = ROOT / str(MANIFEST["paths"]["logs"])
AUDIT = ROOT / str(MANIFEST["paths"]["audit"])
SEEDS = [int(value) for value in MANIFEST["randomness"]["seeds"]]

for directory in (RAW, DERIVED, LOGS, AUDIT):
    directory.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        if not rows:
            raise ValueError(f"fieldnames required for empty CSV: {path}")
        ordered: list[str] = []
        for row in rows:
            for key in row:
                if key not in ordered:
                    ordered.append(key)
        fieldnames = ordered
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials <= 0:
        raise ValueError("trials must be positive")
    phat = successes / trials
    z2 = z * z
    denominator = 1 + z2 / trials
    center = (phat + z2 / (2 * trials)) / denominator
    radius = z * math.sqrt((phat * (1 - phat) + z2 / (4 * trials)) / trials) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def clopper_pearson_upper_zero(trials: int, alpha: float = 0.05) -> float:
    if trials <= 0:
        raise ValueError("trials must be positive")
    return 1 - alpha ** (1 / trials)


def exact_binomial_cdf(length: int, probability: float, threshold_count: int) -> float:
    return sum(
        math.comb(length, k) * probability**k * (1 - probability) ** (length - k)
        for k in range(threshold_count + 1)
    )


class DeterministicSecrets:
    """Test-driver RNG with the subset of secrets used by protocol_v7."""

    def __init__(self, seed: int):
        self._rng = random.Random(seed)

    def token_bytes(self, length: int) -> bytes:
        return bytes(self._rng.getrandbits(8) for _ in range(length))

    def token_hex(self, nbytes: int | None = None) -> str:
        length = 32 if nbytes is None else nbytes
        return self.token_bytes(length).hex()

    def randbits(self, k: int) -> int:
        return self._rng.getrandbits(k)


@contextmanager
def deterministic_protocol_secrets(seed: int) -> Iterator[None]:
    import protocol_v7

    previous = protocol_v7.secrets
    protocol_v7.secrets = DeterministicSecrets(seed)  # type: ignore[assignment]
    try:
        yield
    finally:
        protocol_v7.secrets = previous  # type: ignore[assignment]


def parse_prism_output(text: str) -> dict[str, Any]:
    results = re.findall(r"^Result:\s*([^\r\n]+)", text, flags=re.MULTILINE)
    states = re.findall(r"States:\s*([0-9,]+)", text)
    transitions = re.findall(r"Transitions:\s*([0-9,]+)", text)
    construction = re.findall(r"Time for model construction:\s*([0-9.]+) seconds", text)
    checking = re.findall(r"Time for model checking:\s*([0-9.]+) seconds", text)
    return {
        "results": results,
        "states": int(states[-1].replace(",", "")) if states else None,
        "transitions": int(transitions[-1].replace(",", "")) if transitions else None,
        "construction_seconds": float(construction[-1]) if construction else None,
        "checking_seconds": float(checking[-1]) if checking else None,
    }


def run_prism(
    model: Path,
    props: Path,
    *,
    constants: dict[str, Any] | None,
    log_path: Path,
    exact: bool = True,
    timeout: int = 240,
) -> dict[str, Any]:
    prism = os.environ.get("IAQD_PRISM", r"D:\prism-wrapper\prism.bat")
    command = [prism, str(model), str(props)]
    if constants:
        command += ["-const", ",".join(f"{key}={value}" for key, value in constants.items())]
    if exact:
        command.append("-exact")
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    text = completed.stdout + completed.stderr
    log_path.write_text(
        "COMMAND=" + subprocess.list2cmdline(command) + "\n"
        + f"EXIT_CODE={completed.returncode}\n"
        + text,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(f"PRISM failed ({completed.returncode}); see {log_path}")
    parsed = parse_prism_output(text)
    parsed["command"] = command
    parsed["exit_code"] = completed.returncode
    parsed["log"] = str(log_path.relative_to(ROOT)).replace("\\", "/")
    return parsed


def verify_sha256_manifest(path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        match = re.match(r"^([0-9A-Fa-f]{64})\s+(.+)$", line)
        if not match:
            continue
        expected, relative = match.groups()
        target = ROOT / relative.strip()
        actual = sha256_file(target) if target.exists() else ""
        status = "MATCH" if actual == expected.upper() else "MISSING" if not target.exists() else "MISMATCH"
        rows.append({"manifest": str(path.relative_to(ROOT)).replace("\\", "/"), "path": relative.strip().replace("\\", "/"), "expected_sha256": expected.upper(), "actual_sha256": actual, "status": status})
    return {
        "rows": rows,
        "match": sum(row["status"] == "MATCH" for row in rows),
        "mismatch": sum(row["status"] == "MISMATCH" for row in rows),
        "missing": sum(row["status"] == "MISSING" for row in rows),
    }


def historical_checksum_status(path: Path) -> dict[str, Any]:
    """Match a flattened historical artifact to its unique original-delivery basename."""

    manifest = ROOT / "original" / "experiment_delivery_metadata" / "checksums.sha256"
    candidates: list[tuple[str, str]] = []
    for line in manifest.read_text(encoding="utf-8-sig").splitlines():
        match = re.match(r"^([0-9A-Fa-f]{64})\s+(.+)$", line)
        if match and Path(match.group(2)).name == path.name:
            candidates.append((match.group(1).upper(), match.group(2)))
    actual = sha256_file(path)
    unique_hashes = sorted({item[0] for item in candidates})
    if len(unique_hashes) == 1 and actual == unique_hashes[0]:
        status = "MATCH_ORIGINAL_DELIVERY"
    elif not candidates:
        status = "NO_ORIGINAL_ENTRY"
    elif actual in unique_hashes:
        status = "MATCH_AMBIGUOUS_PATH"
    else:
        status = "MISMATCH_ORIGINAL_DELIVERY"
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "actual_sha256": actual,
        "original_candidate_count": len(candidates),
        "original_candidate_paths": json.dumps([item[1] for item in candidates], ensure_ascii=False),
        "original_candidate_hashes": json.dumps(unique_hashes),
        "status": status,
    }


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def write_command_log(path: Path, command: Sequence[str], exit_code: int, stdout: str, stderr: str = "") -> None:
    path.write_text(
        "COMMAND=" + subprocess.list2cmdline(list(command)) + "\n"
        + f"EXIT_CODE={exit_code}\n"
        + stdout
        + ("\nSTDERR\n" + stderr if stderr else ""),
        encoding="utf-8",
    )
