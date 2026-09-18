from __future__ import annotations

import copy
import inspect
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TESTS = ROOT / "tests"
sys.path[:0] = [str(SRC), str(TESTS)]

from bell_truth_table_v6 import branch_fixture, expected_observations  # noqa: E402
from protocol_v7 import (  # noqa: E402
    BASIC,
    CANON_ID,
    ID_A,
    ID_B,
    RELEASE,
    SPEC_ID,
    IAQDSession,
    ProtocolViolation,
    ce1_decode,
    ce1_encode,
    compute_final_tq,
    parse_record,
    record_with_changes,
    validate_record,
)
from protocol_v7_scenarios import (  # noqa: E402
    _ack_fixture,
    _confirm_fixture,
    _opening_fixture,
    run_ablation_catalog,
    run_attack_catalog,
    short_tag_exhaustive_check,
)
from quantum_common_v6 import BELL_LABELS  # noqa: E402


def finish_release(session: IAQDSession):
    commit_a = session.send_commit_set(ID_A)
    commit_b = session.send_commit_set(ID_B)
    if not session.deliver(commit_a, ID_B).accepted:
        return session.result()
    if not session.deliver(commit_b, ID_A).accepted:
        return session.result()
    for j in range(1, int(session.params["r"]) + 1):
        leader = session.leader_for_block(j)
        follower = ID_B if leader == ID_A else ID_A
        first = session.send_open(leader, j)
        if not session.deliver(first, follower).accepted:
            return session.result()
        second = session.send_open(follower, j)
        if not session.deliver(second, leader).accepted:
            return session.result()
    confirm_b = session.send_confirm(ID_B)
    confirm_a = session.send_confirm(ID_A)
    session.deliver(confirm_b, ID_A)
    session.deliver(confirm_a, ID_B)
    return session.result()


def release_prelude_with_leader(leader: str) -> IAQDSession:
    for _ in range(64):
        session = IAQDSession(mode=RELEASE, message_a="001011", message_b="110100", lambda_bits=4)
        if session.run_prelude() and session.first_leader() == leader:
            return session
    raise AssertionError(f"could not sample first leader {leader}")


def release_ready_with_leader(leader: str) -> IAQDSession:
    session = release_prelude_with_leader(leader)
    commit_a = session.send_commit_set(ID_A)
    commit_b = session.send_commit_set(ID_B)
    assert session.deliver(commit_a, ID_B).accepted
    assert session.deliver(commit_b, ID_A).accepted
    return session


class ProtocolV7Tests(unittest.TestCase):
    def test_01_ce1_is_typed_injective_and_rejects_floats(self) -> None:
        values = [None, False, True, 0, 256, b"x", "x", [1, "1"], {"b": 1, "a": 2}]
        encodings = [ce1_encode(value) for value in values]
        self.assertEqual(len(encodings), len(set(encodings)))
        self.assertEqual([ce1_decode(raw) for raw in encodings], values)
        with self.assertRaises(TypeError):
            ce1_encode(0.1)
        with self.assertRaises(ValueError):
            ce1_decode(ce1_encode("ok") + b"x")

    def test_02_honest_basic_completes_with_bob_first_and_one_auth_per_record(self) -> None:
        session = IAQDSession(mode=BASIC, message_a="0011", message_b="1100")
        result = session.run_honest()
        self.assertTrue(result.success)
        self.assertEqual(result.recovered_by_alice, "1100")
        self.assertEqual(result.recovered_by_bob, "0011")
        self.assertEqual(result.metrics["post_ke_record_count"], 6)
        self.assertEqual(result.metrics["mac_record_count"], 4)
        self.assertEqual(result.metrics["prf_confirm_count"], 2)
        records = [parse_record(raw) for raw in result.wire_records]
        self.assertTrue(all(set(record) == {"hdr", "payload", "auth"} for record in records))
        self.assertTrue(all("record_mac" not in record for record in records))
        dialogue_order = [record["hdr"]["type"] for record in records if record["hdr"]["seq"] == 1]
        self.assertEqual(dialogue_order, ["DIALOGUE_B_QPASS", "DIALOGUE_A"])
        confirms = [record for record in records if record["hdr"]["type"] == "CONFIRM"]
        self.assertTrue(all(record["auth"]["kind"] == "PRF" for record in confirms))
        self.assertTrue(all(set(record["payload"]) == {"message_bits", "confirm_label"} for record in confirms))

    def test_03_honest_release_nondivisible_last_block_completes(self) -> None:
        session = IAQDSession(mode=RELEASE, message_a="001011", message_b="110100", lambda_bits=4)
        result = session.run_honest()
        self.assertTrue(result.success)
        self.assertEqual(session.params["release_lengths"], [4, 2])
        self.assertEqual(result.metrics["post_ke_record_count"], 12)
        self.assertEqual(result.metrics["causal_layers_core_spec"], 10)
        for state in (result.alice_state, result.bob_state):
            self.assertEqual(state["S"], 6)
            self.assertEqual(state["V"], 6)
            self.assertLessEqual(state["max_gap"], 4)

    def test_04_lambda_one_and_full_length_boundaries_complete(self) -> None:
        for lam, expected in ((1, [1] * 6), (6, [6])):
            session = IAQDSession(mode=RELEASE, message_a="001011", message_b="110100", lambda_bits=lam)
            result = session.run_honest()
            self.assertTrue(result.success)
            self.assertEqual(session.params["release_lengths"], expected)
            self.assertLessEqual(result.alice_state["max_gap"], lam)
            self.assertLessEqual(result.bob_state["max_gap"], lam)

    def test_05_both_first_leaders_have_complete_honest_traces(self) -> None:
        seen: dict[str, str] = {}
        for leader in (ID_A, ID_B):
            session = release_prelude_with_leader(leader)
            result = finish_release(session)
            self.assertTrue(result.success)
            seen[leader] = result.sid
        self.assertEqual(set(seen), {ID_A, ID_B})

    def test_06_local_states_are_independent_and_qber_never_appears_at_alice(self) -> None:
        session = IAQDSession(mode=BASIC, message_a="0011", message_b="1100")
        self.assertIsNot(session.alice.params, session.bob.params)
        self.assertIsNot(session.alice.keys, session.bob.keys)
        self.assertIsNot(session.alice.private_local, session.bob.private_local)
        self.assertNotIn("own_mask", session.alice.private_local)
        self.assertTrue(session.send_quantum())
        ack = session.send_qrecv_ack()
        self.assertTrue(session.deliver(ack, ID_A).accepted)
        decoy = session.send_decoy_info()
        self.assertTrue(session.deliver(decoy, ID_B).accepted)
        self.assertTrue(session.bob_detection(errors=0))
        self.assertIn("detection_counts", session.bob.private_local)
        self.assertNotIn("detection_counts", session.alice.private_local)
        self.assertFalse(session.alice.detection_passed)
        session.perform_quantum_core()
        raw = session.send_dialogue_b_qpass()
        self.assertIsNone(session.alice.t_q)
        self.assertFalse(session.alice.qpass_authenticated)
        self.assertTrue(session.deliver(raw, ID_A).accepted)
        self.assertEqual(session.alice.t_q, session.bob.t_q)
        self.assertTrue(session.alice.qpass_authenticated)

    def test_07_validator_has_only_record_and_local_state_inputs(self) -> None:
        parameters = list(inspect.signature(validate_record).parameters)
        self.assertEqual(parameters, ["raw_record", "receiver_state"])
        source = inspect.getsource(validate_record)
        self.assertNotIn("attack_name", source)
        self.assertNotIn("is_malicious", source)
        self.assertNotIn("expected_result", source)

    def test_08_tq_domain_has_no_numeric_qber_and_is_publicly_reconstructed(self) -> None:
        signature = list(inspect.signature(compute_final_tq).parameters)
        self.assertEqual(signature, ["sid", "params_hash", "length_s3", "length_s4", "h_ack", "h_dist"])
        source = inspect.getsource(compute_final_tq)
        self.assertNotIn("measured_qber", source)
        session = IAQDSession(mode=BASIC, message_a="00", message_b="11")
        self.assertTrue(session.run_prelude())
        self.assertEqual(session.alice.t_q, session.bob.t_q)

    def test_09_qrecv_missing_forged_and_replayed_are_local_failures(self) -> None:
        missing = IAQDSession(mode=BASIC, message_a="00", message_b="11")
        missing.send_quantum()
        missing.local_timeout(ID_A, "QRECV_ACK")
        self.assertTrue(missing.alice.terminated)
        self.assertTrue(missing.bob.active)
        with self.assertRaises(ProtocolViolation):
            missing.send_decoy_info()

        forged, raw, receiver = _ack_fixture()
        altered = record_with_changes(raw, {("auth", "tag"): "00" * 32})
        before_seq = forged.alice.next_recv_seq
        result = forged.deliver(altered, receiver)
        self.assertFalse(result.accepted)
        self.assertEqual(forged.alice.next_recv_seq, before_seq)
        self.assertFalse(forged.alice.qrecv_accepted)
        self.assertTrue(forged.bob.active)

        replay, raw, receiver = _ack_fixture()
        self.assertTrue(replay.deliver(raw, receiver).accepted)
        self.assertFalse(replay.deliver(raw, receiver).accepted)

    def test_10_decoy_cannot_be_sent_before_local_ack_acceptance(self) -> None:
        session = IAQDSession(mode=BASIC, message_a="00", message_b="11")
        session.send_quantum()
        session.send_qrecv_ack()
        with self.assertRaisesRegex(ProtocolViolation, "decoy_requires_locally_verified_qrecv"):
            session.send_decoy_info()

    def test_11_mode_downgrade_reflection_replay_order_and_lengths_are_covered(self) -> None:
        attacks = {item.name: item for item in run_attack_catalog()}
        selected = {
            "parameter_replacement",
            "direction_reflection",
            "same_session_replay",
            "future_sequence",
            "qrecv_L3_tamper",
            "wrong_T_Q",
        }
        self.assertTrue(all(attacks[name].passed for name in selected))
        self.assertTrue(all(not attacks[name].accepted for name in selected))

    def test_12_early_and_duplicate_confirmation_are_rejected(self) -> None:
        early = IAQDSession(mode=BASIC, message_a="00", message_b="11")
        with self.assertRaisesRegex(ProtocolViolation, "confirmation_requires_complete_local_recovery"):
            early.send_confirm(ID_A)
        session, raw, receiver = _confirm_fixture()
        self.assertTrue(session.deliver(raw, receiver).accepted)
        duplicate = session.deliver(raw, receiver)
        self.assertFalse(duplicate.accepted)
        self.assertIn(duplicate.reason, {"unexpected_type_or_stage", "replayed_or_old_sequence", "duplicate_record"})

    def test_13_invalid_opening_never_increments_v_and_replay_never_double_counts(self) -> None:
        session, raw, receiver = _opening_fixture()
        state = session.bob if receiver == ID_B else session.alice
        before_v = state.V
        parsed = parse_record(raw)
        parsed["payload"]["chunk"] = ("1" if parsed["payload"]["chunk"][0] == "0" else "0") + parsed["payload"]["chunk"][1:]
        result = session.deliver(ce1_encode(parsed), receiver)
        self.assertFalse(result.accepted)
        self.assertEqual(state.V, before_v)

        replay, raw, receiver = _opening_fixture()
        replay_state = replay.bob if receiver == ID_B else replay.alice
        self.assertTrue(replay.deliver(raw, receiver).accepted)
        accepted_v = replay_state.V
        self.assertFalse(replay.deliver(raw, receiver).accepted)
        self.assertEqual(replay_state.V, accepted_v)

    def test_14_each_honest_party_preserves_bound_when_either_side_leads_then_peer_stops(self) -> None:
        for honest in (ID_A, ID_B):
            corrupt = ID_B if honest == ID_A else ID_A
            for first_leader in (ID_A, ID_B):
                session = release_ready_with_leader(first_leader)
                if first_leader != honest:
                    corrupt_open = session.send_open(corrupt, 1)
                    self.assertTrue(session.deliver(corrupt_open, honest).accepted)
                honest_open = session.send_open(honest, 1)
                honest_state = session.alice if honest == ID_A else session.bob
                self.assertLessEqual(honest_state.S - honest_state.V, 4)
                self.assertLessEqual(honest_state.max_gap, 4)
                session.local_abort(corrupt)
                session.local_timeout(honest, "NEXT_OPEN")
                saved = (honest_state.S, honest_state.V, honest_state.max_gap, len(session.wire_records))
                with self.assertRaises(ProtocolViolation):
                    session.send_open(honest, 2)
                self.assertEqual((honest_state.S, honest_state.V, honest_state.max_gap, len(session.wire_records)), saved)

    def test_15_mismatched_mask_is_not_misclassified_as_release_invariant_failure(self) -> None:
        session = IAQDSession(mode=RELEASE, message_a="001011", message_b="110100", lambda_bits=4)
        self.assertTrue(session.run_prelude())
        original_mask = session.bob.private_local["own_mask"]
        session.bob.private_local["own_mask"] = ("1" if original_mask[0] == "0" else "0") + original_mask[1:]
        commit_a = session.send_commit_set(ID_A)
        commit_b = session.send_commit_set(ID_B)
        self.assertTrue(session.deliver(commit_a, ID_B).accepted)
        self.assertTrue(session.deliver(commit_b, ID_A).accepted)
        for j in range(1, int(session.params["r"]) + 1):
            leader = session.leader_for_block(j)
            follower = ID_B if leader == ID_A else ID_A
            first = session.send_open(leader, j)
            self.assertTrue(session.deliver(first, follower).accepted)
            second = session.send_open(follower, j)
            self.assertTrue(session.deliver(second, leader).accepted)
        self.assertNotEqual(session.alice.recovered_peer_message, session.bob.own_message())
        self.assertLessEqual(session.alice.max_gap, 4)
        self.assertEqual((session.alice.S, session.alice.V), (6, 6))
        confirm_a = session.send_confirm(ID_A)
        self.assertTrue(session.deliver(confirm_a, ID_B).accepted)
        confirm_b = session.send_confirm(ID_B)
        result = session.deliver(confirm_b, ID_A)
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "authentication_tag_invalid")

    def test_16_detection_failure_is_not_teleported_to_alice(self) -> None:
        session = IAQDSession(mode=BASIC, message_a="00", message_b="11")
        session.send_quantum()
        ack = session.send_qrecv_ack()
        self.assertTrue(session.deliver(ack, ID_A).accepted)
        decoy = session.send_decoy_info()
        self.assertTrue(session.deliver(decoy, ID_B).accepted)
        self.assertFalse(session.bob_detection(errors=1, trials=2))
        self.assertTrue(session.bob.terminated)
        self.assertTrue(session.alice.active)
        self.assertIsNone(session.alice.t_q)
        session.local_timeout(ID_A, "DIALOGUE_B_QPASS")
        self.assertTrue(session.alice.terminated)

    def test_17_new_sessions_have_fresh_sid_key_handles_and_masks(self) -> None:
        message_a = "01" * 32
        message_b = "10" * 32
        first = IAQDSession(mode=RELEASE, message_a=message_a, message_b=message_b, lambda_bits=16)
        second = IAQDSession(mode=RELEASE, message_a=message_a, message_b=message_b, lambda_bits=16)
        self.assertTrue(first.run_prelude())
        self.assertTrue(second.run_prelude())
        self.assertNotEqual(first.sid, second.sid)
        self.assertNotEqual(first.alice.keys.key_handle, second.alice.keys.key_handle)
        self.assertNotEqual(first.alice.private_local["own_mask"], second.alice.private_local["own_mask"])
        self.assertNotEqual(first.bob.private_local["own_mask"], second.bob.private_local["own_mask"])

    def test_18_all_42_legacy_attacks_are_migrated_to_actual_checks(self) -> None:
        outcomes = run_attack_catalog()
        self.assertEqual(len(outcomes), 42)
        self.assertEqual(len({item.name for item in outcomes}), 42)
        self.assertTrue(all(item.passed and not item.accepted and item.reason for item in outcomes))
        self.assertTrue(all(item.validation_trace for item in outcomes))

    def test_19_nine_legacy_ablations_are_mapped_without_forcing_failure(self) -> None:
        outcomes = run_ablation_catalog()
        self.assertEqual(len(outcomes), 9)
        by_name = {item.legacy_name: item for item in outcomes}
        self.assertEqual(by_name["no_mac"].migration_status, "NOT_APPLICABLE_AS_STATED")
        self.assertEqual(by_name["public_message_hash_confirmation"].migration_status, "ORIGINAL_AQD_ONLY")
        self.assertTrue(all(item.evidence and item.interpretation for item in outcomes))

    def test_20_short_tag_enumeration_calls_actual_prf_verifier(self) -> None:
        result = short_tag_exhaustive_check()
        self.assertEqual(result["candidate_count"], 256)
        self.assertEqual(result["actual_validator_calls_reaching_prf"], 256)
        self.assertEqual(result["accepted_count"], 1)
        self.assertEqual(result["reason_counts"]["authentication_tag_invalid"], 255)
        self.assertIn("not a preset probability", result["method"])

    def test_21_all_128_independent_bell_branches_recover_correctly(self) -> None:
        checked = 0
        for cluster in ("phi1", "phi2"):
            for message_a in BELL_LABELS:
                for message_b in BELL_LABELS:
                    for observed in expected_observations(cluster, message_a, message_b):
                        block = branch_fixture(cluster, message_a, message_b, observed)
                        session = IAQDSession(mode=BASIC, message_a=message_a, message_b=message_b, quantum_blocks=[block])
                        result = session.run_honest()
                        self.assertTrue(result.success)
                        self.assertEqual(result.recovered_by_alice, message_b)
                        self.assertEqual(result.recovered_by_bob, message_a)
                        checked += 1
        self.assertEqual(checked, 128)

    def test_22_event_log_has_local_before_after_evidence_without_secret_values(self) -> None:
        session = IAQDSession(mode=RELEASE, message_a="001011", message_b="110100", lambda_bits=4)
        result = session.run_honest()
        send_or_deliver = [event for event in result.event_log if event["event"] in {"RECORD_SEND", "RECORD_DELIVER_VERIFY"}]
        self.assertTrue(send_or_deliver)
        self.assertTrue(all("state_before" in event and "state_after" in event for event in send_or_deliver))
        serialized = json.dumps(result.event_log, ensure_ascii=False, sort_keys=True)
        for forbidden in (session.alice.private_local["own_mask"], session.bob.private_local["own_mask"]):
            self.assertNotIn(forbidden, serialized)
        self.assertNotIn("detection_counts\": [", serialized)

    def test_23_unknown_fields_and_wrong_lengths_do_not_partially_accept(self) -> None:
        session, raw, receiver = _ack_fixture()
        record = parse_record(raw)
        record["hdr"]["unknown"] = "x"
        before = copy.deepcopy(session.alice.snapshot())
        result = session.deliver(ce1_encode(record), receiver)
        self.assertFalse(result.accepted)
        self.assertEqual(session.alice.next_recv_seq, before["next_recv_seq"])
        self.assertEqual(session.alice.accepted_tokens, set())
        self.assertFalse(session.alice.qrecv_accepted)

    def test_24_record_headers_bind_reviewed_version_canon_mode_and_parameters(self) -> None:
        result = IAQDSession(mode=BASIC, message_a="00", message_b="11").run_honest()
        for raw in result.wire_records:
            header = parse_record(raw)["hdr"]
            self.assertEqual(header["spec_id"], SPEC_ID)
            self.assertEqual(header["canon_id"], CANON_ID)
            self.assertEqual(header["mode"], BASIC)
            self.assertEqual(len(header["params_hash"]), 64)


if __name__ == "__main__":
    unittest.main(verbosity=2)
