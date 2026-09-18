"""Small, deterministic-in-outcome regression scenarios for protocol_v7.

Scenario names and classifications live here, outside the receiving validator.
The validator is never passed a scenario name, expected result, or trust label.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import hmac
import sys
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from protocol_v7 import (  # noqa: E402
    BASIC,
    ID_A,
    ID_B,
    RELEASE,
    IAQDSession,
    ProtocolViolation,
    ValidationResult,
    _build_record,
    ce1_decode,
    ce1_encode,
    parse_record,
    record_with_changes,
)


@dataclasses.dataclass
class ScenarioOutcome:
    name: str
    legacy_source: str
    category: str
    accepted: bool
    reason: str
    passed: bool
    interpretation: str
    validation_trace: list[str]
    event_log: list[dict[str, Any]]


@dataclasses.dataclass
class AblationOutcome:
    legacy_name: str
    migration_status: str
    observed_v7: str
    evidence: str
    interpretation: str


def _flip_bits(bits: str) -> str:
    return ("1" if bits[0] == "0" else "0") + bits[1:]


def _outcome(
    name: str,
    session: IAQDSession,
    result: ValidationResult | None = None,
    *,
    reason: str | None = None,
    category: str = "external_record",
    interpretation: str = "The reviewed local check rejected the altered action.",
) -> ScenarioOutcome:
    if result is not None:
        accepted = result.accepted
        actual_reason = result.reason
        trace = result.trace
    else:
        accepted = False
        actual_reason = str(reason)
        trace = [f"local_send_gate:{actual_reason}"]
    return ScenarioOutcome(
        name=name,
        legacy_source="e5_registered_attacks_v6.csv",
        category=category,
        accepted=accepted,
        reason=actual_reason,
        passed=not accepted,
        interpretation=interpretation,
        validation_trace=list(trace),
        event_log=copy.deepcopy(session.event_log),
    )


def _deliver_change(
    name: str,
    fixture: Callable[[], tuple[IAQDSession, bytes, str]],
    changes: dict[tuple[str, ...], Any],
    **metadata: Any,
) -> ScenarioOutcome:
    session, raw, receiver = fixture()
    changed = record_with_changes(raw, changes)
    return _outcome(name, session, session.deliver(changed, receiver), **metadata)


def _ack_fixture() -> tuple[IAQDSession, bytes, str]:
    session = IAQDSession(mode=BASIC, message_a="0011", message_b="1100")
    assert session.send_quantum()
    return session, session.send_qrecv_ack(), ID_A


def _decoy_fixture() -> tuple[IAQDSession, bytes, str]:
    session, ack, _ = _ack_fixture()
    assert session.deliver(ack, ID_A).accepted
    return session, session.send_decoy_info(), ID_B


def _dialogue_b_fixture() -> tuple[IAQDSession, bytes, str]:
    session, decoy, _ = _decoy_fixture()
    assert session.deliver(decoy, ID_B).accepted
    assert session.bob_detection()
    session.perform_quantum_core()
    return session, session.send_dialogue_b_qpass(), ID_A


def _dialogue_a_fixture() -> tuple[IAQDSession, bytes, str]:
    session, dialogue_b, _ = _dialogue_b_fixture()
    assert session.deliver(dialogue_b, ID_A).accepted
    return session, session.send_dialogue_a(), ID_B


def _confirm_fixture(*, conf_tag_bits: int = 128) -> tuple[IAQDSession, bytes, str]:
    session = IAQDSession(mode=BASIC, message_a="0011", message_b="1100", conf_tag_bits=conf_tag_bits)
    assert session.run_prelude()
    return session, session.send_confirm(ID_A), ID_B


def _commit_fixture() -> tuple[IAQDSession, bytes, str]:
    session = IAQDSession(mode=RELEASE, message_a="001011", message_b="110100", lambda_bits=4)
    assert session.run_prelude()
    return session, session.send_commit_set(ID_A), ID_B


def _opening_fixture() -> tuple[IAQDSession, bytes, str]:
    session = IAQDSession(mode=RELEASE, message_a="001011", message_b="110100", lambda_bits=4)
    assert session.run_prelude()
    commit_a = session.send_commit_set(ID_A)
    commit_b = session.send_commit_set(ID_B)
    assert session.deliver(commit_a, ID_B).accepted
    assert session.deliver(commit_b, ID_A).accepted
    leader = session.leader_for_block(1)
    return session, session.send_open(leader, 1), ID_B if leader == ID_A else ID_A


def _tag_zero(raw: bytes) -> bytes:
    record = parse_record(raw)
    record["auth"]["tag"] = "00" * (len(record["auth"]["tag"]) // 2)
    return ce1_encode(record)


def _remove_auth_tag(raw: bytes) -> bytes:
    record = parse_record(raw)
    del record["auth"]["tag"]
    return ce1_encode(record)


def _change_decoy_basis(raw: bytes) -> bytes:
    record = parse_record(raw)
    disclosure = record["payload"]["D_decoy"]
    current = disclosure["bases_S3"]["values"][0]
    disclosure["bases_S3"]["values"][0] = "X" if current == "Z" else "Z"
    return ce1_encode(record)


def _gate_decoy_before_ack() -> ScenarioOutcome:
    session = IAQDSession(mode=BASIC, message_a="00", message_b="11")
    session.send_quantum()
    try:
        session.send_decoy_info()
    except ProtocolViolation as error:
        return _outcome("decoy_before_verified_qrecv", session, reason=error.reason, category="local_send_gate")
    raise AssertionError("early decoy unexpectedly sent")


def _gate_confirmation_before_recovery() -> ScenarioOutcome:
    session = IAQDSession(mode=BASIC, message_a="00", message_b="11")
    try:
        session.send_confirm(ID_A)
    except ProtocolViolation as error:
        return _outcome("confirmation_before_recovery", session, reason=error.reason, category="local_send_gate")
    raise AssertionError("early confirmation unexpectedly sent")


def _replay(name: str, fixture: Callable[[], tuple[IAQDSession, bytes, str]]) -> ScenarioOutcome:
    session, raw, receiver = fixture()
    assert session.deliver(raw, receiver).accepted
    return _outcome(name, session, session.deliver(raw, receiver), category="replay")


def _wrong_local_recovery() -> ScenarioOutcome:
    session, raw, receiver = _confirm_fixture()
    state = session.bob
    state.recovered_peer_message = _flip_bits(str(state.recovered_peer_message))
    return _outcome(
        "wrong_locally_recovered_message",
        session,
        session.deliver(raw, receiver),
        category="local_fault",
        interpretation="A local recovery mismatch is detected by I8; it is not modeled as a network attacker label.",
    )


def _opening_out_of_order() -> ScenarioOutcome:
    session, _raw, receiver = _opening_fixture()
    sender = session.alice if receiver == ID_B else session.bob
    opening = copy.deepcopy(sender.private_local["own_openings"][2])
    opening["commit_set_hash"] = sender.public_hashes["own_commit_set_hash"]
    raw = _build_record(sender, "OPEN", opening, str(sender.t_q), 4)
    return _outcome("out_of_order_opening", session, session.deliver(raw, receiver))


def _duplicate_opening() -> ScenarioOutcome:
    session, raw, receiver = _opening_fixture()
    assert session.deliver(raw, receiver).accepted
    return _outcome("duplicate_opening", session, session.deliver(raw, receiver), category="replay")


def _decision_replacement() -> ScenarioOutcome:
    session, raw, receiver = _dialogue_b_fixture()
    changed = record_with_changes(raw, {("payload", "decision"): "QFAIL"})
    return _outcome("final_T_Q_decision_replacement", session, session.deliver(changed, receiver))


def _payload_tamper() -> ScenarioOutcome:
    session, raw, receiver = _dialogue_a_fixture()
    parsed = parse_record(raw)
    parsed["payload"]["ciphertext"] = _flip_bits(parsed["payload"]["ciphertext"])
    return _outcome("payload_tamper", session, session.deliver(ce1_encode(parsed), receiver))


def _qber_field_injection() -> ScenarioOutcome:
    session, raw, receiver = _dialogue_b_fixture()
    record = parse_record(raw)
    record["payload"]["qber"] = 0
    changed = ce1_encode(record)
    return _outcome(
        "final_T_Q_qber_replacement",
        session,
        session.deliver(changed, receiver),
        interpretation="V7 has no transmitted QBER field; injecting one is a schema error, not a T_Q update.",
    )


def run_attack_catalog() -> list[ScenarioOutcome]:
    """Execute all 42 legacy attack names against the reviewed V7 interfaces."""

    outcomes: list[ScenarioOutcome] = []
    outcomes.append(_payload_tamper())
    session, raw, receiver = _dialogue_a_fixture()
    outcomes.append(_outcome("record_mac_tamper", session, session.deliver(_tag_zero(raw), receiver), interpretation="Legacy name retained for mapping; V7 rejects the sole ordinary-record MAC tag and has no outer record_mac."))
    outcomes.append(_deliver_change("protocol_version_replacement", _dialogue_a_fixture, {("hdr", "spec_id"): "IAQD-V6"}))
    outcomes.append(_deliver_change("sid_replacement", _dialogue_a_fixture, {("hdr", "sid"): "00" * 32}))
    outcomes.append(_deliver_change("sender_replacement", _dialogue_a_fixture, {("hdr", "sender"): ID_B}))
    outcomes.append(_deliver_change("receiver_replacement", _dialogue_a_fixture, {("hdr", "receiver"): ID_A}))
    outcomes.append(_deliver_change("direction_reflection", _dialogue_a_fixture, {("hdr", "direction"): "B2A"}))
    outcomes.append(_deliver_change("type_replacement", _dialogue_a_fixture, {("hdr", "type"): "CONFIRM"}))
    outcomes.append(_deliver_change("cross_session_migration", _dialogue_a_fixture, {("hdr", "sid"): "11" * 32}))
    outcomes.append(_replay("same_session_replay", _dialogue_a_fixture))
    outcomes.append(_deliver_change("old_sequence", _dialogue_a_fixture, {("hdr", "seq"): 0}))
    outcomes.append(_deliver_change("future_sequence", _dialogue_a_fixture, {("hdr", "seq"): 2}))
    outcomes.append(_gate_decoy_before_ack())

    session, raw, receiver = _decoy_fixture()
    outcomes.append(_outcome("decoy_wrong_K_dist", session, session.deliver(_tag_zero(raw), receiver)))
    session, raw, receiver = _decoy_fixture()
    outcomes.append(_outcome("D_decoy_tamper", session, session.deliver(_change_decoy_basis(raw), receiver)))
    session, raw, receiver = _decoy_fixture()
    outcomes.append(_outcome("tau_dist_tamper", session, session.deliver(_tag_zero(raw), receiver), interpretation="Legacy tau_dist is V7's sole K_dist auth.tag, not an inner-plus-outer pair."))
    session, raw, receiver = _decoy_fixture()
    outcomes.append(_outcome("tau_dist_missing", session, session.deliver(_remove_auth_tag(raw), receiver)))
    outcomes.append(_deliver_change("decoy_cross_session_migration", _decoy_fixture, {("hdr", "sid"): "22" * 32}))
    outcomes.append(_deliver_change("decoy_direction_reflection", _decoy_fixture, {("hdr", "direction"): "B2A"}))
    outcomes.append(_replay("decoy_record_replay", _decoy_fixture))

    outcomes.append(_deliver_change("qrecv_L3_tamper", _ack_fixture, {("payload", "L3"): 999}))
    outcomes.append(_deliver_change("qrecv_L4_tamper", _ack_fixture, {("payload", "L4"): 999}))
    outcomes.append(_deliver_change("qrecv_identity_swap", _ack_fixture, {("hdr", "sender"): ID_A, ("hdr", "receiver"): ID_B}))
    outcomes.append(_deliver_change("qrecv_direction_reflection", _ack_fixture, {("hdr", "direction"): "A2B"}))
    session, raw, receiver = _ack_fixture()
    outcomes.append(_outcome("qrecv_forged_tau_ack", session, session.deliver(_tag_zero(raw), receiver)))
    session, raw, receiver = _ack_fixture()
    outcomes.append(_outcome("qrecv_wrong_ack_key", session, session.deliver(_tag_zero(raw), receiver)))

    outcomes.append(_deliver_change("wrong_T_Q", _dialogue_a_fixture, {("hdr", "qctx"): "33" * 32}))
    outcomes.append(_deliver_change("pre_quantum_context_replacement", _ack_fixture, {("hdr", "qctx"): "44" * 32}))
    outcomes.append(_qber_field_injection())
    outcomes.append(_deliver_change("final_T_Q_threshold_replacement", _dialogue_b_fixture, {("hdr", "params_hash"): "55" * 32}))
    outcomes.append(_decision_replacement())
    outcomes.append(_deliver_change("parameter_replacement", _dialogue_a_fixture, {("hdr", "mode"): RELEASE}))
    outcomes.append(_gate_confirmation_before_recovery())

    session, raw, receiver = _confirm_fixture()
    outcomes.append(_outcome("wrong_gamma", session, session.deliver(_tag_zero(raw), receiver)))
    outcomes.append(_wrong_local_recovery())

    session, raw, receiver = _opening_fixture()
    parsed = parse_record(raw)
    parsed["payload"]["chunk"] = _flip_bits(parsed["payload"]["chunk"])
    outcomes.append(_outcome("wrong_commitment_opening", session, session.deliver(ce1_encode(parsed), receiver)))
    session, raw, receiver = _commit_fixture()
    parsed = parse_record(raw)
    parsed["payload"]["commitments"][0]["d_j"] = "66" * 32
    outcomes.append(_outcome("commitment_replacement", session, session.deliver(ce1_encode(parsed), receiver)))
    outcomes.append(_duplicate_opening())
    outcomes.append(_opening_out_of_order())
    session, raw, receiver = _opening_fixture()
    parsed = parse_record(raw)
    parsed["payload"]["salt"] = "77" * 16
    outcomes.append(_outcome("salt_mismatch", session, session.deliver(ce1_encode(parsed), receiver)))
    outcomes.append(_deliver_change("block_number_replacement", _opening_fixture, {("payload", "j"): 2}))
    outcomes.append(_deliver_change("cross_session_opening", _opening_fixture, {("hdr", "sid"): "88" * 32}))
    if len(outcomes) != 42:
        raise AssertionError(f"legacy attack migration count changed unexpectedly: {len(outcomes)}")
    return outcomes


def run_ablation_catalog(attacks: list[ScenarioOutcome] | None = None) -> list[AblationOutcome]:
    """Map nine V6 ablations without making weakened modes selectable in V7."""

    by_name = {item.name: item for item in (attacks or run_attack_catalog())}
    mapped = [
        AblationOutcome("no_sid", "MAPPED_TO_REQUIRED_CHECK", "sid replacement rejected", by_name["sid_replacement"].reason, "sid is also inside the prescribed authenticator, so neighboring checks may be redundant."),
        AblationOutcome("no_direction", "MAPPED_TO_REQUIRED_CHECK", "reflection rejected", by_name["direction_reflection"].reason, "direction is checked locally and authenticated; removing one layer need not imply acceptance."),
        AblationOutcome("no_sequence", "MAPPED_TO_REQUIRED_CHECK", "replay rejected", by_name["same_session_replay"].reason, "single-accept tokens and the exact next sequence jointly protect replay."),
        AblationOutcome("no_receipt_ack", "MAPPED_TO_SEND_GATE", "early decoy blocked", by_name["decoy_before_verified_qrecv"].reason, "the sender-side ACK gate is a reviewed causal precondition."),
        AblationOutcome("no_mac", "NOT_APPLICABLE_AS_STATED", "legacy outer record_mac removed", by_name["record_mac_tamper"].reason, "V7 has no generic outer record_mac; ordinary records still have their one specified direction MAC."),
        AblationOutcome("no_keyed_confirmation", "MAPPED_TO_REQUIRED_CHECK", "wrong confirmation rejected", by_name["wrong_gamma"].reason, "I8 is protected only by the confirmation PRF and is not double wrapped."),
        AblationOutcome("no_K_dist_validation", "MAPPED_TO_REQUIRED_CHECK", "decoy alteration rejected", by_name["D_decoy_tamper"].reason, "DECOY_INFO uses the sole K_dist tag after schema/context checks."),
        AblationOutcome("public_record_hash", "LEGACY_OPTION_REMOVED", "not selectable", "params bind HMAC-SHA256-REFERENCE", "A public hash is not an authentication alternative in the reviewed parameter set."),
        AblationOutcome("public_message_hash_confirmation", "ORIGINAL_AQD_ONLY", "not a revised-protocol option", "no public message hash field", "The original-AQD dictionary oracle remains a cryptanalysis case, not a V7 I8 mode."),
    ]
    if len(mapped) != 9:
        raise AssertionError("legacy ablation mapping is incomplete")
    return mapped


def short_tag_exhaustive_check() -> dict[str, Any]:
    """Try every 8-bit tag through validate_record for one fixed I8 context."""

    session, raw, _receiver = _confirm_fixture(conf_tag_bits=8)
    receiver_template = copy.deepcopy(session.bob)
    record = parse_record(raw)
    accepted_tags: list[str] = []
    verifier_calls = 0
    reasons: dict[str, int] = {}
    for candidate in range(256):
        trial_record = copy.deepcopy(record)
        trial_record["auth"]["tag"] = f"{candidate:02x}"
        trial_state = copy.deepcopy(receiver_template)
        result = __import__("protocol_v7").validate_record(ce1_encode(trial_record), trial_state)
        verifier_calls += int("verify_auth:PRF" in result.trace)
        reasons[result.reason] = reasons.get(result.reason, 0) + 1
        if result.accepted:
            accepted_tags.append(f"{candidate:02x}")
    return {
        "tag_bits": 8,
        "candidate_count": 256,
        "actual_validator_calls_reaching_prf": verifier_calls,
        "accepted_count": len(accepted_tags),
        "accepted_tags": accepted_tags,
        "reason_counts": reasons,
        "method": "exhaustive fixed-context calls to protocol_v7.validate_record; not a preset probability",
        "event_log": session.event_log,
    }
