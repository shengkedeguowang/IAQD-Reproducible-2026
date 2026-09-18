"""Read-only integrity checks for the curated IAQD reference evidence."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_RUN = "experiment_release_v7_20260905T203845_CST"
PRISM_RUN = "20260905T200413_CST"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def select_one(rows: list[dict[str, str]], **expected: str) -> dict[str, str]:
    matches = [row for row in rows if all(row.get(key) == value for key, value in expected.items())]
    if len(matches) != 1:
        raise AssertionError(f"expected one row for {expected}, found {len(matches)}")
    return matches[0]


def main() -> int:
    checks: list[str] = []
    experiment_derived = ROOT / "results" / "derived" / EXPERIMENT_RUN
    experiment_raw = ROOT / "results" / "raw" / EXPERIMENT_RUN
    prism_derived = ROOT / "results" / "derived" / "prism_release_v7" / PRISM_RUN

    complete = read_csv(experiment_derived / "c_complete_noisy_protocol_aggregate_release_v7.csv")
    basic = select_one(complete, mode="BASIC", p="0.1", metric="full_protocol_acceptance")
    release = select_one(complete, mode="RELEASE", p="0.1", metric="full_protocol_acceptance")
    assert basic["successes"] == "2738" and basic["trials"] == "5000" and basic["estimate"] == "0.5476"
    assert release["successes"] == "2756" and release["trials"] == "5000" and release["estimate"] == "0.5512"
    checks.append("complete noisy-protocol acceptance anchors")

    branches = read_csv(experiment_raw / "c_quantum_v7_128_branches_release_v7.csv")
    assert len(branches) == 128 and all(row["v7_success"] == "True" for row in branches)
    checks.append("128 V7 Bell branches")

    resources = read_csv(experiment_raw / "g_resource_accounting_release_v7.csv")
    release_resource = select_one(resources, scheme="IAQD_RELEASE_V7", n_blocks="32", ell="4", lambda_bits="4")
    assert release_resource["actual_serialized_post_ke_bytes"] == "34631"
    checks.append("RELEASE n=32 resource anchor")

    grid = read_csv(prism_derived / "grid_results.csv")
    assert len(grid) == 52 and all(float(row["release_violation_pmax"]) == 0.0 for row in grid)
    checks.append("52 normal PRISM configurations")

    summary = json.loads((prism_derived / "summary.json").read_text(encoding="utf-8"))
    assert summary["validation_failures"] == [] and summary["negative_relaxed"]["release_violation_pmax"] == "1"
    checks.append("PRISM negative-control sensitivity")

    print("Reference evidence check passed:")
    for check in checks:
        print(f"  - {check}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, FileNotFoundError, KeyError, ValueError) as error:
        print(f"Reference evidence check failed: {error}", file=sys.stderr)
        raise SystemExit(1)
