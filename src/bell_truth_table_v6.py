from __future__ import annotations

from quantum_common_v6 import xor_label


# Independent paper association table.  It intentionally does not import the
# analytic expansion or the circuit implementation.
_ALICE_LOOKUP = {
    "phi1": {("00", "01"): "00", ("00", "10"): "01", ("11", "01"): "10", ("11", "10"): "11"},
    "phi2": {("01", "01"): "00", ("01", "10"): "01", ("10", "01"): "10", ("10", "10"): "11"},
}
_BOB_LOOKUP = {
    "phi1": {("00", "00"): "01", ("00", "01"): "10", ("11", "10"): "01", ("11", "11"): "10"},
    "phi2": {("01", "00"): "01", ("01", "01"): "10", ("10", "10"): "01", ("10", "11"): "10"},
}


def expected_observations(cluster_type: str, alice_message: str, bob_message: str) -> set[tuple[str, str, str]]:
    observations: set[tuple[str, str, str]] = set()
    for (label12, base56), base34 in _ALICE_LOOKUP[cluster_type].items():
        observations.add((label12, xor_label(base34, bob_message), xor_label(base56, alice_message)))
    return observations


def recover_messages(
    cluster_type: str,
    observed: tuple[str, str, str],
    *,
    known_alice: str,
    known_bob: str,
) -> tuple[str | None, str | None, dict[str, str]]:
    label12, label34, label56 = observed
    base56 = xor_label(label56, known_alice)
    base34_for_alice = _ALICE_LOOKUP[cluster_type].get((label12, base56))
    recovered_bob = None if base34_for_alice is None else xor_label(base34_for_alice, label34)
    base34 = xor_label(label34, known_bob)
    base56_for_bob = _BOB_LOOKUP[cluster_type].get((label12, base34))
    recovered_alice = None if base56_for_bob is None else xor_label(base56_for_bob, label56)
    intermediates = {
        "label12": label12,
        "M_B": base34,
        "tilde_M_B": label34,
        "M_A": base56,
        "tilde_M_A": label56,
        "alice_base56": base56,
        "alice_base34": base34_for_alice or "INVALID",
        "bob_base34": base34,
        "bob_base56": base56_for_bob or "INVALID",
    }
    return recovered_alice, recovered_bob, intermediates


def branch_fixture(cluster_type: str, alice_message: str, bob_message: str, observed: tuple[str, str, str]) -> dict[str, object]:
    recovered_alice, recovered_bob, intermediates = recover_messages(
        cluster_type, observed, known_alice=alice_message, known_bob=bob_message
    )
    return {
        "cluster_type": cluster_type,
        "bell_labels": list(observed),
        "decoded_intermediates": intermediates,
        "decoded_for_alice": recovered_bob,
        "decoded_for_bob": recovered_alice,
    }
