from __future__ import annotations

from qiskit import QuantumCircuit, transpile


# Independently reviewed Clifford synthesis.  This module contains no analytic
# Bell expansion or protocol decoding table.
_COMMON = (
    ("h", 2), ("h", 3), ("h", 5),
    ("cx", 3, 2), ("cx", 5, 3), ("cx", 3, 4), ("h", 4), ("cx", 4, 3),
    ("s", 1), ("h", 1), ("s", 1), ("s", 5), ("h", 5),
    ("swap_cx", 5, 1), ("cx", 1, 5), ("h", 2), ("s", 4), ("h", 4),
    ("swap_cx", 2, 4), ("cx", 2, 4), ("cx", 1, 2),
    ("s", 0), ("h", 0), ("s", 0), ("s", 4), ("h", 4),
    ("swap_cx", 1, 0), ("cx", 4, 1), ("cx", 0, 1), ("h", 4), ("s", 4),
    ("swap_cx", 0, 4), ("cx", 4, 0), ("s", 4), ("h", 4), ("s", 4),
)
_FINAL = {
    "phi1": (("x", 0), ("y", 1), ("x", 2), ("x", 4), ("x", 5)),
    "phi2": (("y", 1), ("x", 2), ("x", 4), ("x", 5)),
}
FORBIDDEN_OPERATIONS = {"initialize", "state_preparation", "unitary", "isometry"}


def _apply(circuit: QuantumCircuit, instruction: tuple[object, ...]) -> None:
    name, *raw_qubits = instruction
    qubits = [int(q) for q in raw_qubits]
    if name == "swap_cx":
        circuit.cx(qubits[0], qubits[1])
        circuit.cx(qubits[1], qubits[0])
        circuit.cx(qubits[0], qubits[1])
    elif name in {"h", "s", "x", "y", "z"}:
        getattr(circuit, str(name))(qubits[0])
    elif name == "cx":
        circuit.cx(qubits[0], qubits[1])
    else:
        raise ValueError(name)


def explicit_cluster_circuit(cluster_type: str) -> QuantumCircuit:
    if cluster_type not in _FINAL:
        raise ValueError(cluster_type)
    circuit = QuantumCircuit(6, name=f"explicit_{cluster_type}")
    for instruction in _COMMON + _FINAL[cluster_type]:
        _apply(circuit, instruction)
    assert_no_state_injection(circuit)
    return circuit


def assert_no_state_injection(circuit: QuantumCircuit) -> None:
    """Recursively inspect source, decomposed, and transpiled circuits."""
    variants = [circuit, circuit.decompose(reps=10)]
    variants.append(transpile(circuit, basis_gates=["h", "s", "x", "y", "z", "cx"], optimization_level=0))
    for variant in variants:
        bad = {item.operation.name.lower() for item in variant.data} & FORBIDDEN_OPERATIONS
        if bad:
            raise AssertionError(f"state-injection operation(s): {sorted(bad)}")


def apply_pauli_label(circuit: QuantumCircuit, qubit: int, label: str) -> None:
    if label[1] == "1":
        circuit.x(qubit)
    if label[0] == "1":
        circuit.z(qubit)


def full_protocol_circuit(cluster_type: str, alice_message: str, bob_message: str, *, measure: bool) -> QuantumCircuit:
    circuit = QuantumCircuit(6, 6) if measure else QuantumCircuit(6)
    circuit.compose(explicit_cluster_circuit(cluster_type), inplace=True)
    apply_pauli_label(circuit, 2, bob_message)
    apply_pauli_label(circuit, 4, alice_message)
    if measure:
        for first, second in ((0, 1), (2, 3), (4, 5)):
            circuit.cx(first, second)
            circuit.h(first)
            circuit.measure(first, first)
            circuit.measure(second, second)
    assert_no_state_injection(circuit)
    return circuit
