# IAQD Core Analysis

This cleaned package retains only the material needed to study:

- the AQD protocol and its security weaknesses;
- the IAQD protocol improvement and its authenticated transcript;
- security analysis, attack regression, quantum simulation, and finite-state
  model checking; and
- quantum, classical-communication, authentication, and round-resource
  accounting.

Removed material includes duplicated source inputs, the original manuscript
attachments and PDF, patch/review material, audit histories, change logs,
failure logs, report snapshots, checksums that only described those artifacts,
and duplicate LaTeX copies. The retained manuscript is
`paper/iaqd_analysis.tex`.

## Retained layout

- `paper/` — the single retained manuscript source for AQD/IAQD analysis.
- `src/` — AQD and IAQD protocol, quantum-core, security, and resource code.
- `experiments/` — fixed-parameter drivers for the retained analyses.
- `prism/` and `tools/` — DTMC/MDP models and PRISM runner.
- `tests/` — semantic and attack-regression tests.
- `results/` — raw and derived outputs for the retained analyses.
- `scripts/quick_check.ps1` — read-only reference-data and V7 semantic check.

## Requirements and check

The reference environment uses Python 3.12, Qiskit 2.5.2, Qiskit Aer 0.17.2,
and PRISM 4.10.1. Python dependencies are pinned in `requirements.txt`.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
.\scripts\quick_check.ps1
```

Install PRISM separately if you need to rerun the finite-state models. The
stored result data are preserved; full experiment drivers may regenerate
outputs, so run them in a disposable working copy.
