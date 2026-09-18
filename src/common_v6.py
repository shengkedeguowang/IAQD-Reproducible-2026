from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from fractions import Fraction
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parent
OUTPUTS = EXP / "outputs"
RAW = OUTPUTS / "raw"
PROCESSED = OUTPUTS / "processed"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"
COUNTEREXAMPLES = OUTPUTS / "counterexamples"
LOGS = OUTPUTS / "logs"
REPORTS = EXP / "reports"
PRISM_DIR = EXP / "prism"
CONFIG = EXP / "configs" / "runtime_v6.yaml"
MANIFEST = EXP / "configs" / "experiment_manifest_v6.yaml"
PRISM = Path(os.environ.get("PRISM_WRAPPER", r"D:\prism-wrapper\prism.bat"))


def ensure_dirs() -> None:
    for path in (RAW, PROCESSED, FIGURES, TABLES, COUNTEREXAMPLES, LOGS, REPORTS, PRISM_DIR):
        path.mkdir(parents=True, exist_ok=True)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def runtime_config() -> dict[str, Any]:
    return load_yaml(CONFIG)


def seeds() -> list[int]:
    values = [int(value) for value in runtime_config()["seeds"]]
    if len(values) != 30 or len(set(values)) != 30:
        raise ValueError("runtime_v6.yaml must contain 30 distinct seeds")
    return values


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in materialized:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    if trials <= 0:
        return math.nan, math.nan
    z = 1.959963984540054
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def clopper_pearson_upper_zero(trials: int, alpha: float = 0.05) -> float:
    return 1 - alpha ** (1 / trials) if trials > 0 else math.nan


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * float(p_values[index])))
        adjusted[index] = running
    return adjusted


def summary_row(experiment: str, configuration: str, metric: str, value: Any, *, ci_low: Any = "", ci_high: Any = "", unit: str = "", status: str = "complete", raw_hash: str = "", notes: str = "", seed: str = "") -> dict[str, Any]:
    return {
        "experiment": experiment,
        "configuration": configuration,
        "seed": seed,
        "metric": metric,
        "value": value,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "unit": unit,
        "status": status,
        "raw_file_sha256": raw_hash,
        "notes": notes,
    }


def parse_prism_output(text: str) -> dict[str, Any]:
    result_matches = re.findall(r"^Result:\s*([^\s]+)", text, flags=re.MULTILINE)
    states_match = re.search(r"States:\s*([0-9,]+)", text)
    transitions_match = re.search(r"Transitions:\s*([0-9,]+)", text)
    build_match = re.search(r"Time for model construction:\s*([0-9.]+)", text)
    check_match = re.search(r"Time for model checking:\s*([0-9.]+)", text)
    value: float | str | None = None
    if result_matches:
        token = result_matches[-1]
        try:
            value = float(token)
        except ValueError:
            try:
                value = float(Fraction(token))
            except (ValueError, ZeroDivisionError):
                value = token
    return {
        "result": value,
        "states": int(states_match.group(1).replace(",", "")) if states_match else None,
        "transitions": int(transitions_match.group(1).replace(",", "")) if transitions_match else None,
        "construction_seconds": float(build_match.group(1)) if build_match else None,
        "checking_seconds": float(check_match.group(1)) if check_match else None,
    }


def run_prism(model: Path, props: Path, *, prop: int = 1, constants: dict[str, Any] | None = None, log_path: Path, strategy_path: Path | None = None, states_path: Path | None = None, transitions_path: Path | None = None, exact: bool = False, explicit: bool = False, timeout: int = 180) -> dict[str, Any]:
    if not PRISM.exists():
        raise FileNotFoundError(f"PRISM wrapper not found at {PRISM}")
    arguments = [str(PRISM), str(model), str(props), "-prop", str(prop)]
    if constants:
        arguments += ["-const", ",".join(f"{key}={value}" for key, value in constants.items())]
    if exact:
        arguments.append("-exact")
    if explicit:
        arguments.append("-explicit")
    export_moves: list[tuple[Path, Path]] = []

    def short_export_path(destination: Path, kind: str) -> Path:
        token = hashlib.sha256(f"{destination}|{kind}".encode("utf-8")).hexdigest()[:12]
        temporary = Path(tempfile.gettempdir()) / f"aqd_v6_{token}_{kind}.txt"
        if temporary.exists():
            temporary.unlink()
        export_moves.append((temporary, destination))
        return temporary

    if strategy_path is not None:
        strategy_path.parent.mkdir(parents=True, exist_ok=True)
        arguments += ["-exportstrat", str(short_export_path(strategy_path, "strategy"))]
    if states_path is not None:
        states_path.parent.mkdir(parents=True, exist_ok=True)
        arguments += ["-exportstates", str(short_export_path(states_path, "states"))]
    if transitions_path is not None:
        transitions_path.parent.mkdir(parents=True, exist_ok=True)
        arguments += ["-exporttrans", str(short_export_path(transitions_path, "transitions"))]
    started = time.perf_counter()
    completed = subprocess.run(
        ["cmd.exe", "/d", "/c", *arguments],
        cwd=str(PRISM_DIR),
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    elapsed = time.perf_counter() - started
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"PRISM failed with code {completed.returncode}; see {log_path}")
    missing_exports = []
    for temporary, destination in export_moves:
        if temporary.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary.replace(destination)
        else:
            missing_exports.append(str(destination))
    if missing_exports:
        raise RuntimeError(f"PRISM did not create required exports {missing_exports}; see {log_path}")
    parsed = parse_prism_output(completed.stdout)
    if parsed["result"] is None:
        raise RuntimeError(f"PRISM produced no result; see {log_path}")
    parsed.update({"wall_seconds": elapsed, "command": subprocess.list2cmdline(arguments), "returncode": 0})
    return parsed


def exact_binomial_cdf(length: int, probability: float, threshold: float) -> float:
    from scipy.stats import binom

    return float(binom.cdf(math.floor(length * threshold), length, probability))


def environment_snapshot(source_pdf: Path | None = None) -> dict[str, Any]:
    import importlib.metadata as metadata
    import psutil

    packages: dict[str, str | None] = {}
    for name in ("qiskit", "qiskit-aer", "numpy", "scipy", "PyYAML", "matplotlib", "pytest"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    snapshot = {
        "captured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version,
        "executable": sys.executable,
        "packages": packages,
        "operating_system": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "memory_total_bytes": psutil.virtual_memory().total,
        "seeds": seeds(),
    }
    if source_pdf is not None:
        snapshot["source_pdf"] = {"path": str(source_pdf), "sha256": sha256_file(source_pdf)}
    return snapshot


ensure_dirs()

