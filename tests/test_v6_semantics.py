from __future__ import annotations

import ast
import copy
import csv
import hashlib
import hmac
import inspect
import json
import math
from pathlib import Path

import numpy as np
import pytest
from qiskit.quantum_info import Statevector, state_fidelity

from bell_truth_table_v6 import branch_fixture, expected_observations, recover_messages
from cluster_analytic_v6 import analytic_cluster_target
from cluster_circuit_v6 import assert_no_state_injection, explicit_cluster_circuit, full_protocol_circuit
from protocol_v6 import (
    GAMMA_FIELD_NAMES,
    PROTOCOL_VERSION,
    IAQDReferenceExecutor,
    ReceiverState,
    ValidationPolicy,
    assert_quantum_block_semantics,
    build_complete_decoy_disclosure,
    canonical_fields,
    compute_commitment,
    compute_final_tq,
    compute_gamma,
    compute_qrecv_ack,
    compute_tau_dist,
    complete_decoy_disclosure_is_valid,
    decoy_auth_header,
    compute_sid,
    create_record,
    derive_session_keys,
    mutate_record,
    parse_record,
    record_auth_material,
    validate_record,
)
from quantum_common_v6 import BELL_LABELS
from run_e2_v6 import simulate_seed
from run_e3_v6 import probe_unitary
from run_e6_v6 import simulate_error_counts
from run_e5_v6 import KEYS as E5_KEYS, SID as E5_SID, decoy_fixture


EXP = Path(__file__).resolve().parents[1]
SRC = EXP / "src"
RAW = EXP / "outputs" / "raw"
COUNTEREXAMPLES = EXP / "outputs" / "counterexamples"
REPORTS = EXP / "reports"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_01_sid_is_sensitive_to_every_paper_field() -> None:
    values = [PROTOCOL_VERSION, "Alice", "Bob", b"A" * 16, b"B" * 16, {"mode": "basic", "n": 1}]
    baseline = compute_sid(*values)
    variants = [
        ["IAQD-old", *values[1:]],
        [values[0], "Mallory", *values[2:]],
        [*values[:2], "Mallory", *values[3:]],
        [*values[:3], b"C" * 16, *values[4:]],
        [*values[:4], b"D" * 16, values[5]],
        [*values[:5], {"mode": "fair", "n": 1}],
    ]
    assert all(compute_sid(*variant) != baseline for variant in variants)


def test_02_ack_key_is_domain_separated_and_no_commit_keys_exist() -> None:
    keys = derive_session_keys(b"K" * 32, "11" * 32, 2)
    values = [keys.mac_a, keys.mac_b, keys.conf_a, keys.conf_b, keys.otp_a, keys.otp_b, keys.ack, keys.dist]
    assert len({value for value in values}) == len(values)
    assert not hasattr(keys, "commit_a") and not hasattr(keys, "commit_b")


def _qrecv_fixture() -> tuple[ReceiverState, object, bytes]:
    sid = compute_sid(PROTOCOL_VERSION, "Alice", "Bob", b"A" * 16, b"B" * 16, {"mode": "basic"})
    keys = derive_session_keys(b"S" * 32, sid, 1)
    pre = hashlib.sha256(b"pre").hexdigest()
    state = ReceiverState(
        sid=sid, protocol_version=PROTOCOL_VERSION, local_identity="Alice", peer_identity="Bob",
        receive_direction="B2A", params={"mode": "basic"}, pre_quantum_context_digest=pre,
        expected_record_type="QRECV",
    )
    payload = {"recv": "QRECV", "L3": 5, "L4": 7}
    payload["tau_ack"] = compute_qrecv_ack(keys.ack, sid, "Bob", "Alice", "QRECV", 5, 7)
    raw = create_record(
        sid=sid, sender="Bob", receiver="Alice", direction="B2A", record_type="QRECV", seq=0,
        params={"mode": "basic"}, quantum_transcript_digest=pre, payload=payload, session_keys=keys,
    )
    return state, keys, raw


def test_03_qrecv_exactly_binds_sid_identities_recv_l3_l4_and_ack_key() -> None:
    state, keys, raw = _qrecv_fixture()
    assert validate_record(raw, state, keys).accepted
    for field, value in (("recv", "OTHER"), ("L3", 6), ("L4", 8)):
        state, keys, raw = _qrecv_fixture()
        parsed = parse_record(raw)
        parsed["payload"][field] = value
        forged = json.dumps(parsed, sort_keys=True, separators=(",", ":")).encode()
        assert not validate_record(forged, state, keys).accepted


def test_04_decoy_disclosure_requires_verified_qrecv_state() -> None:
    state, keys, _ = _qrecv_fixture()
    state.local_identity, state.peer_identity, state.receive_direction = "Bob", "Alice", "A2B"
    state.expected_record_type = "DECOY_DISCLOSURE"
    raw = create_record(
        sid=state.sid, sender="Alice", receiver="Bob", direction="A2B", record_type="DECOY_DISCLOSURE",
        seq=0, params=state.params, quantum_transcript_digest=state.pre_quantum_context_digest,
        payload={"qrecv_record_id": "none", "decoy_basis_digest": "x"}, session_keys=keys,
    )
    result = validate_record(raw, state, keys)
    assert not result.accepted and result.reason == "qrecv_not_verified"


def _tq_fields() -> dict[str, object]:
    return {
        "proto_ver": PROTOCOL_VERSION, "sid": "ab" * 32, "ID_A": "Alice", "ID_B": "Bob",
        "L3": 5, "L4": 5, "qrecv_record_id": "q", "qrecv_digest": "1" * 64,
        "decoy_record_digest": "2" * 64, "measured_qber": 0.01, "qber_threshold": 0.1,
        "decision": "accept", "params": {"mode": "basic"},
    }


def test_05_final_tq_is_field_sensitive_and_frozen_once() -> None:
    fields = _tq_fields()
    baseline = compute_final_tq(fields)
    for key in fields:
        changed = dict(fields)
        changed[key] = ({"changed": True} if key == "params" else (0.02 if key == "measured_qber" else f"changed-{key}"))
        assert compute_final_tq(changed) != baseline
    state = ReceiverState(
        sid="ab" * 32, protocol_version=PROTOCOL_VERSION, local_identity="Alice", peer_identity="Bob",
        receive_direction="B2A", params={}, pre_quantum_context_digest="0" * 64,
    )
    state.freeze_final_quantum_transcript(baseline)
    with pytest.raises(RuntimeError):
        state.freeze_final_quantum_transcript(baseline)
    with pytest.raises(AttributeError):
        state.final_quantum_transcript_digest = "0" * 64
    assert state.final_quantum_transcript_digest == baseline


def test_06_qber_reject_cannot_enter_dialogue() -> None:
    result = IAQDReferenceExecutor(7, measured_qber=0.2, qber_threshold=0.1).run_basic("00", "11")
    events = [item["event"] for item in result.trace]
    assert result.terminal_state == "ABORT_QBER" and not result.success
    assert "consume_external_bell_decode" not in events and "protected_dialogue_record_A" not in events


def test_07_main_commitment_is_unkeyed_and_salt_bound() -> None:
    source = inspect.getsource(compute_commitment)
    assert "hmac.new" not in source and "hashlib.sha3_256" in source
    assert "key" not in inspect.signature(compute_commitment).parameters
    salt = b"s" * 16
    value = compute_commitment("ab" * 32, "Alice", 0, "10", salt)
    assert value != compute_commitment("ab" * 32, "Alice", 0, "10", b"t" * 16)


def test_08_three_e4_truth_sources_have_no_forbidden_imports() -> None:
    forbidden = {
        "cluster_circuit_v6.py": {"cluster_analytic_v6", "bell_truth_table_v6"},
        "cluster_analytic_v6.py": {"cluster_circuit_v6", "bell_truth_table_v6"},
        "bell_truth_table_v6.py": {"cluster_circuit_v6", "cluster_analytic_v6"},
    }
    for name, blocked in forbidden.items():
        tree = ast.parse((SRC / name).read_text(encoding="utf-8"))
        imports = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        assert not (imports & blocked)


def test_09_circuits_recursively_contain_no_state_injection() -> None:
    for cluster in ("phi1", "phi2"):
        assert_no_state_injection(explicit_cluster_circuit(cluster))
        assert_no_state_injection(full_protocol_circuit(cluster, "11", "10", measure=False))


def test_10_explicit_circuits_match_independent_analytic_targets() -> None:
    for cluster in ("phi1", "phi2"):
        assert state_fidelity(Statevector.from_instruction(explicit_cluster_circuit(cluster)), Statevector(analytic_cluster_target(cluster))) >= 1 - 1e-12


def test_11_all_nonzero_bell_branches_decode_with_independent_table() -> None:
    count = 0
    for cluster in ("phi1", "phi2"):
        for alice in BELL_LABELS:
            for bob in BELL_LABELS:
                observed_set = expected_observations(cluster, alice, bob)
                assert len(observed_set) == 4
                for observed in observed_set:
                    recovered_alice, recovered_bob, intermediates = recover_messages(cluster, observed, known_alice=alice, known_bob=bob)
                    assert recovered_alice == alice and recovered_bob == bob
                    assert intermediates["M_B"] == ("".join(str(int(x) ^ int(y)) for x, y in zip(observed[1], bob)))
                    assert intermediates["tilde_M_B"] == observed[1]
                    assert intermediates["M_A"] == ("".join(str(int(x) ^ int(y)) for x, y in zip(observed[2], alice)))
                    assert intermediates["tilde_M_A"] == observed[2]
                    count += 1
    assert count == 128


def test_12_mutating_bell_branch_changes_classical_quantum_input() -> None:
    observed = sorted(expected_observations("phi1", "00", "11"))[0]
    first = branch_fixture("phi1", "00", "11", observed)
    mutated_labels = (observed[0], observed[1], "00" if observed[2] != "00" else "11")
    second = branch_fixture("phi1", "00", "11", mutated_labels)
    assert first["bell_labels"] != second["bell_labels"]
    assert first["decoded_intermediates"] != second["decoded_intermediates"] or first["decoded_for_bob"] != second["decoded_for_bob"]


def test_13_e4_128_rows_record_actual_labels_and_executor_inputs() -> None:
    data = rows(RAW / "e4_basic_end_to_end_128_branches_v6.csv")
    assert len(data) == 128 and all(row["branch_success"] == "True" for row in data)
    for row in data:
        passed = json.loads(row["passed_classical_quantum_input"])
        assert passed["bell_labels"] == [row["bell_branch_12"], row["bell_branch_34"], row["bell_branch_56"]]


def test_14_timeout_counterexample_contains_explicit_timeout_and_bob_abort() -> None:
    value = json.loads((COUNTEREXAMPLES / "e1_abort_terminal_shortest_counterexample_v6.json").read_text(encoding="utf-8"))
    actions = [step["action"] for step in value["trace"] if "action" in step]
    assert actions == ["send_cb", "alice_recovers_b", "alice_abort", "bob_waiting_for_ca", "timeout", "bob_abort"]
    assert value["trace"][-1]["state"]["terminal"] is True


def test_15_no_timeout_model_does_not_invent_bob_abort() -> None:
    value = json.loads((COUNTEREXAMPLES / "e1_abort_no_timeout_liveness_witness_v6.json").read_text(encoding="utf-8"))
    final = value["trace"][-1]["state"]
    assert final["BobWaitingForCA"] and not final["terminal"] and not final["OutB_abort"]
    assert float(value["prism_terminal_bob_abort_property"]["result"]) == 0.0


def test_16_e2_uses_particle_events_and_expected_one_particle_rates() -> None:
    source = (SRC / "run_e2_v6.py").read_text(encoding="utf-8")
    assert "rng.binomial(trials, p_und)" not in source
    random_imr = simulate_seed(100, 200000, 5000, 1, 1.0, "random_basis_imr", 0.5, 0.8)["undetected_trials"] / 200000
    replacement = simulate_seed(101, 200000, 5000, 1, 1.0, "random_replacement", 0.5, 0.8)["undetected_trials"] / 200000
    assert abs(random_imr - 0.75) < 0.01 and abs(replacement - 0.5) < 0.01


def test_17_u_theta_probe_is_unitary() -> None:
    for theta in np.linspace(0, np.pi / 2, 17):
        unitary = probe_unitary(float(theta))
        assert np.max(np.abs(unitary.conj().T @ unitary - np.eye(4))) < 1e-12


def test_18_gamma_has_only_paper_fields_and_no_direction_or_sequence() -> None:
    assert GAMMA_FIELD_NAMES == ("sid", "role_label", "message", "ciphertext_a", "ciphertext_b", "T_Q")
    source = inspect.getsource(compute_gamma)
    assert '"direction"' not in source and '"seq"' not in source


def test_19_outer_record_mac_binds_direction_and_sequence() -> None:
    sid = "ab" * 32
    keys = derive_session_keys(b"K" * 32, sid, 1)
    common = dict(ver=PROTOCOL_VERSION, sid=sid, sender="Alice", receiver="Bob", direction="A2B", type="X", seq=0, params={}, T_Q="0" * 64)
    changed_direction = {**common, "direction": "B2A"}
    changed_sequence = {**common, "seq": 1}
    payload = {"x": 1}
    assert record_auth_material(common, payload) != record_auth_material(changed_direction, payload)
    assert record_auth_material(common, payload) != record_auth_material(changed_sequence, payload)


def test_20_confirmation_payload_is_strict_and_requires_local_context() -> None:
    result = IAQDReferenceExecutor(9).run_basic("00", "11")
    confirmations = [parse_record(raw) for raw in result.wire_records if parse_record(raw)["hdr"]["type"] == "MESSAGE_CONFIRMATION"]
    assert confirmations and all(set(item["payload"]) == {"confirmation"} for item in confirmations)
    state, keys, raw = _qrecv_fixture()
    state.local_identity, state.peer_identity, state.receive_direction = "Bob", "Alice", "A2B"
    state.expected_record_type = "MESSAGE_CONFIRMATION"
    state.freeze_final_quantum_transcript(state.pre_quantum_context_digest)
    gamma = compute_gamma("A", keys.conf_a, state.sid, b"A", b"ca", b"cb", state.final_quantum_transcript_digest)
    raw = create_record(
        sid=state.sid, sender="Alice", receiver="Bob", direction="A2B", record_type="MESSAGE_CONFIRMATION", seq=0,
        params=state.params, quantum_transcript_digest=state.final_quantum_transcript_digest,
        payload={"confirmation": gamma}, session_keys=keys,
    )
    assert validate_record(raw, state, keys).reason == "confirmation_context_unavailable"


def test_21_validator_has_exact_three_inputs_and_no_attack_label() -> None:
    assert list(inspect.signature(validate_record).parameters) == ["raw_record_bytes", "receiver_state", "session_keys"]
    source = inspect.getsource(validate_record).lower()
    assert "attack" not in source and "scenario" not in source and "expected_result" not in source


def test_22_complete_validator_rejects_every_registered_attack() -> None:
    data = rows(RAW / "e5_registered_attacks_v6.csv")
    required = {"qrecv_L3_tamper", "qrecv_L4_tamper", "qrecv_wrong_ack_key", "protocol_version_replacement", "commitment_replacement", "cross_session_opening"}
    assert required <= {row["attack"] for row in data}
    assert data and all(row["accepted"] == "False" for row in data)
    assert all(row["original_raw_record_hex"] and row["mutated_raw_record_hex"] for row in data)


def test_23_ablations_preserve_real_or_bounded_negative_results() -> None:
    data = rows(RAW / "e5_ablation_search_v6.csv")
    statuses = {row["status"] for row in data}
    assert statuses <= {"REAL_ERRONEOUS_ACCEPTANCE", "NO_COUNTEREXAMPLE_WITHIN_BOUND"}
    assert "REAL_ERRONEOUS_ACCEPTANCE" in statuses


def test_24_fair_ciphertext_commitment_opening_share_same_mask() -> None:
    result = IAQDReferenceExecutor(10).run_fair("0011", "1100", 1, cluster_sequence=["phi1", "phi2"])
    assert result.success
    assert "".join(item["chunk"] for item in result.commitments["A"]) == result.masks["R_A"]
    assert "".join(item["chunk"] for item in result.commitments["B"]) == result.masks["R_B"]
    recovery = next(item["index"] for item in result.trace if item["event"] == "fair_message_recovery_complete")
    confirmation = next(item["index"] for item in result.trace if item["event"].startswith("fair_keyed_confirmation"))
    assert confirmation > recovery


def test_25_cluster_sequence_is_consumed_and_changes_execution() -> None:
    a = IAQDReferenceExecutor(11).run_fair("0011", "1100", 1, cluster_sequence=["phi1", "phi1"])
    b = IAQDReferenceExecutor(11).run_fair("0011", "1100", 1, cluster_sequence=["phi2", "phi2"])
    assert [x["cluster_type"] for x in a.quantum_inputs] != [x["cluster_type"] for x in b.quantum_inputs]
    assert a.sid != b.sid


def test_26_final_opening_chunk_uses_true_remaining_length() -> None:
    result = IAQDReferenceExecutor(12).run_fair("001111", "110000", 4, cluster_sequence=["phi1", "phi2", "phi1"])
    assert len(result.commitments["A"][-1]["chunk"]) == 2
    assert max(prefix["information_lead"] for prefix in result.fairness_prefixes) <= 4


def test_27_every_mdp_configuration_has_all_k_queries_and_no_target_guard() -> None:
    model = (EXP / "prism" / "iaqd_fair_protocol_v6.nm").read_text(encoding="utf-8")
    transitions = model.split("label ", 1)[0]
    assert "fair_violation" not in transitions and "maximum_lead" not in transitions and "abs(L_A-L_B)" not in transitions
    data = rows(RAW / "e6_mdp_all_information_lead_levels_v6.csv")
    grouped: dict[str, list[int]] = {}
    for row in data:
        grouped.setdefault(row["configuration"], []).append(int(row["k"]))
    assert grouped
    for config, ks in grouped.items():
        m = int(next(row["message_length_bits"] for row in data if row["configuration"] == config))
        assert sorted(ks) == list(range(m + 1))


def test_28_claimed_witness_policies_are_nonempty_parseable_and_reach_target() -> None:
    policy_files = sorted(COUNTEREXAMPLES.glob("e6_*_reconstructed_witness_policy_v6.json"))
    assert policy_files
    for path in policy_files:
        value = json.loads(path.read_text(encoding="utf-8"))
        assert value["kind"] == "reconstructed_witness_policy"
        assert value["entries"] and value["verified_final_lead"] == value["target_maximum_information_lead"]


def test_29_dtmc_monte_carlo_generates_l_events() -> None:
    source = inspect.getsource(simulate_error_counts)
    assert "rng.random((current, length)" in source and "binomial" not in source
    counts = simulate_error_counts(13, 17, 0.2, 1000, 100)
    assert len(counts) == 1000 and np.all(counts <= 17)


def test_30_e7_resources_and_performance_come_from_real_executor() -> None:
    resource = rows(RAW / "e7_reference_implementation_resources_v6.csv")
    performance = rows(RAW / "e7_reference_performance_by_seed_v6.csv")
    assert resource and performance
    assert all(int(row["actual_total_wire_bytes"]) > int(row["actual_decoy_record_bytes"]) > int(row["actual_decoy_payload_bytes"]) for row in resource)
    assert all(row["full_D_decoy_in_wire_record"] == "True" and row["tau_dist_covers_full_D_decoy"] == "True" for row in resource)
    assert all(int(row["upper_layer_round_count"]) == (5 if row["mode"] == "basic" else 6 + int(np.ceil((2 * int(row["n_blocks"])) / int(row["lambda_bits"])))) for row in resource)
    counts: dict[tuple[str, str, str, str], int] = {}
    for row in performance:
        key = (row["mode"], row["n_blocks"], row["lambda_bits"], row["ell_per_sequence"])
        counts[key] = counts.get(key, 0) + 1
    assert counts and all(value == 30 for value in counts.values())


def test_31_v6_source_never_reads_v5_results_as_measurements() -> None:
    combined = "\n".join(path.read_text(encoding="utf-8") for path in SRC.glob("*.py"))
    forbidden = ("AQD_E1-E7_实验交付包_v5_20260830/outputs", "AQD_E1-E7_实验交付包_v5_20260830\\outputs")
    assert forbidden[0] not in combined and forbidden[1] not in combined
    assert "read_csv" not in inspect.getsource(__import__("run_all_v6").zip_member_hash)


def test_32_protected_v5_zip_and_paper_hashes_are_unchanged() -> None:
    value = json.loads((REPORTS / "INPUT_PROVENANCE_V6.json").read_text(encoding="utf-8"))
    assert value["input_mutation_status"] == "UNCHANGED"
    assert value["inputs"]["v5_zip"]["sha256_at_start"].lower() == "950a5e86744e0313a84e75939018f4a29b9e21b3e7802ed9548e6f284c28fe88"
    assert value["inputs"]["v5_zip"]["sha256_at_start"] == value["end_verification"]["v5_zip"]["sha256"]
    assert value["inputs"]["paper_pdf"]["sha256_at_start"] == value["end_verification"]["paper_pdf"]["sha256"]


def test_33_executor_checks_all_128_branch_intermediates_and_ciphertexts() -> None:
    checked = 0
    for cluster in ("phi1", "phi2"):
        for alice in BELL_LABELS:
            for bob in BELL_LABELS:
                for observed in expected_observations(cluster, alice, bob):
                    block = branch_fixture(cluster, alice, bob, observed)
                    result = IAQDReferenceExecutor(1000 + checked).run_basic(alice, bob, quantum_blocks=[block])
                    event = next(item for item in result.trace if item["event"] == "consume_external_bell_decode")
                    assert event["M_B"] == block["decoded_intermediates"]["M_B"]
                    assert event["tilde_M_B"] == observed[1]
                    assert event["M_A"] == block["decoded_intermediates"]["M_A"]
                    assert event["tilde_M_A"] == observed[2]
                    assert result.success and result.terminal_state == "COMPLETED"
                    checked += 1
    assert checked == 128


def test_34_swapping_M_A_and_M_B_is_detected_by_independent_algebra() -> None:
    observed = next(
        item
        for item in expected_observations("phi1", "00", "11")
        if branch_fixture("phi1", "00", "11", item)["decoded_intermediates"]["M_A"]
        != branch_fixture("phi1", "00", "11", item)["decoded_intermediates"]["M_B"]
    )
    block = branch_fixture("phi1", "00", "11", observed)
    values = block["decoded_intermediates"]
    values["M_A"], values["M_B"] = values["M_B"], values["M_A"]
    with pytest.raises(AssertionError):
        assert_quantum_block_semantics(block, "00", "11")


def test_35_final_T_Q_is_immutable_from_source_and_classical_state_changes() -> None:
    fields = _tq_fields()
    digest = compute_final_tq(fields)
    state = ReceiverState(
        sid="ab" * 32,
        protocol_version=PROTOCOL_VERSION,
        local_identity="Alice",
        peer_identity="Bob",
        receive_direction="B2A",
        params={},
        pre_quantum_context_digest="0" * 64,
    )
    state.freeze_final_quantum_transcript(digest)
    fields["params"] = {"mutated": True}
    state.phase = "CONFIRMATION"
    state.expected_sequence["B2A"] = 99
    state.confirmation_accepted = True
    assert state.final_quantum_transcript_digest == digest


def test_36_final_T_Q_domain_is_deterministic_and_session_sensitive() -> None:
    fields = _tq_fields()
    assert compute_final_tq(fields) == compute_final_tq(dict(fields))
    changed = dict(fields)
    changed["sid"] = "cd" * 32
    assert compute_final_tq(changed) != compute_final_tq(fields)
    source = inspect.getsource(compute_final_tq)
    assert "FINAL_TQ_DOMAIN" in source


def test_37_tau_dist_honest_path_uses_K_dist_and_exact_schema() -> None:
    state, raw = decoy_fixture()
    parsed = parse_record(raw)
    assert set(parsed["payload"]) == {"qrecv_record_id", "D_decoy", "tau_dist"}
    assert validate_record(raw, state, E5_KEYS).accepted
    assert E5_KEYS.counters.counts["tau_dist_verify_calls"] >= 1


def test_38_tau_dist_wrong_key_and_tamper_are_rejected() -> None:
    wrong = derive_session_keys(hashlib.sha256(b"wrong-dist-test").digest(), E5_SID, 1)
    state, raw = decoy_fixture(dist_key=wrong.dist)
    assert validate_record(raw, state, E5_KEYS).reason == "tau_dist_invalid"
    state, raw = decoy_fixture()
    forged = mutate_record(raw, ("payload", "tau_dist"), "00" * 32, E5_KEYS)
    assert validate_record(forged, state, E5_KEYS).reason == "tau_dist_invalid"


def test_39_tau_dist_binds_header_and_decoy_disclosure() -> None:
    state, raw = decoy_fixture()
    parsed = parse_record(raw)
    hdr_dist = decoy_auth_header(parsed["hdr"], parsed["payload"]["qrecv_record_id"])
    baseline = compute_tau_dist(E5_KEYS.dist, hdr_dist, parsed["payload"]["D_decoy"])
    changed_header = dict(hdr_dist)
    changed_header["seq"] = 1
    changed_decoy = copy.deepcopy(parsed["payload"]["D_decoy"])
    changed_decoy["states_S3"]["values"][0] ^= 1
    assert compute_tau_dist(E5_KEYS.dist, changed_header, parsed["payload"]["D_decoy"]) != baseline
    assert compute_tau_dist(E5_KEYS.dist, hdr_dist, changed_decoy) != baseline


def test_40_only_removing_K_dist_validation_accepts_malformed_tau() -> None:
    policy = ValidationPolicy(require_dist_auth=False)
    state, raw = decoy_fixture(policy=policy)
    forged = mutate_record(raw, ("payload", "tau_dist"), "00" * 32, E5_KEYS)
    assert validate_record(forged, state, E5_KEYS).accepted


def test_41_round_schedule_distinguishes_records_from_rounds() -> None:
    basic = IAQDReferenceExecutor(41).run_basic("0011", "1100")
    fair = IAQDReferenceExecutor(42).run_fair("001111", "110000", 4)
    assert basic.metrics["upper_layer_round_count"] == 5
    assert fair.metrics["upper_layer_round_count"] == 6 + int(np.ceil(6 / 4))
    assert basic.metrics["one_way_authenticated_record_count"] != basic.metrics["upper_layer_round_count"]
    assert fair.metrics["one_way_authenticated_record_count"] != fair.metrics["upper_layer_round_count"]


def test_42_runtime_counters_cover_every_derived_subkey_and_K_dist() -> None:
    result = IAQDReferenceExecutor(43).run_basic("00", "11")
    metrics = result.metrics
    assert metrics["KDF_derivation_count"] == 8
    assert set(metrics["KDF_purpose_labels"]) == {
        "IAQD/mac/A", "IAQD/mac/B", "IAQD/conf/A", "IAQD/conf/B",
        "IAQD/otp/A", "IAQD/otp/B", "IAQD/ack/B2A", "IAQD/dist/A2B",
    }
    assert metrics["tau_dist_generation_count"] == metrics["tau_dist_verification_count"] == 1
    assert metrics["QRECV_ack_MAC_generation_count"] == metrics["QRECV_ack_MAC_verification_count"] == 1


def test_43_resource_source_does_not_alias_record_count_as_communication_rounds() -> None:
    protocol_source = (SRC / "protocol_v6.py").read_text(encoding="utf-8")
    e7_source = (SRC / "run_e7_v6.py").read_text(encoding="utf-8")
    assert '"communication_rounds"' not in protocol_source
    assert '"communication_rounds"' not in e7_source
    assert "one_way_authenticated_record_count" in e7_source
    assert "upper_layer_round_count" in e7_source


def test_44_e6_runtime_grid_includes_q0_zero_and_48_dtmc_configurations() -> None:
    import yaml

    config = yaml.safe_load((EXP / "configs" / "runtime_v6.yaml").read_text(encoding="utf-8"))["e6"]
    assert config["honest_qber_values"] == [0.0, 0.01, 0.03, 0.05]
    assert len(config["dtmc_lengths"]) * len(config["honest_qber_values"]) * len(config["qber_threshold_values"]) == 48


def test_45_e6_dtmc_output_has_exact_paper_grid_and_q0_zero_for_every_L_threshold() -> None:
    data = rows(RAW / "e6_dtmc_analytic_prism_mc_v6.csv")
    assert len(data) == 48
    assert {int(row["L"]) for row in data} == {16, 32, 64, 128}
    assert {float(row["q0"]) for row in data} == {0.0, 0.01, 0.03, 0.05}
    assert {float(row["qth"]) for row in data} == {0.05, 0.10, 0.15}
    zero_pairs = {(int(row["L"]), float(row["qth"])) for row in data if float(row["q0"]) == 0.0}
    assert zero_pairs == {(length, threshold) for length in (16, 32, 64, 128) for threshold in (0.05, 0.10, 0.15)}


def test_46_q0_zero_analytic_prism_and_monte_carlo_boundary_is_exact() -> None:
    aggregate = [row for row in rows(RAW / "e6_dtmc_analytic_prism_mc_v6.csv") if float(row["q0"]) == 0.0]
    raw = [row for row in rows(RAW / "e6_dtmc_event_level_by_seed_v6.csv") if float(row["q0"]) == 0.0 and row["role"] == "honest"]
    assert aggregate and len(raw) == 12 * 30
    assert all(float(row["honest_acceptance_analytic"]) == 1.0 for row in aggregate)
    assert all(float(row["honest_false_abort_analytic"]) == 0.0 for row in aggregate)
    assert all(float(row["honest_acceptance_prism"]) == 1.0 for row in aggregate)
    assert all(int(row["accepted"]) == int(row["trials"]) and float(row["acceptance_rate"]) == 1.0 for row in raw)


def test_47_q0_zero_rows_have_finite_wilson_and_p_values() -> None:
    data = [row for row in rows(RAW / "e6_dtmc_analytic_prism_mc_v6.csv") if float(row["q0"]) == 0.0]
    numeric = ("honest_acceptance_ci_low", "honest_acceptance_ci_high", "honest_p_value", "honest_holm_adjusted_p")
    assert all(row[field] not in {"", "nan", "NaN"} and np.isfinite(float(row[field])) for row in data for field in numeric)


def test_48_e6_paper_main_mdp_has_48_configurations_and_full_coverage() -> None:
    data = rows(RAW / "e6_protocol_fairness_mdp_v6.csv")
    assert len(data) == 48 and all(row["configuration_class"] == "paper_main" for row in data)
    assert {int(row["message_length_bits"]) for row in data} == {8, 16, 32}
    assert {int(row["n_blocks"]) for row in data} == {4, 8, 16}
    assert {int(row["lambda_bits"]) for row in data} == {1, 2, 4, 8}
    assert {row["corrupted_party"] for row in data} == {"alice", "bob"}
    assert {row["initial_leader"] for row in data} == {"alice", "bob"}


def test_49_lambda8_covers_both_corrupt_parties_and_both_leaders() -> None:
    data = [row for row in rows(RAW / "e6_protocol_fairness_mdp_v6.csv") if int(row["lambda_bits"]) == 8]
    assert {(row["corrupted_party"], row["initial_leader"]) for row in data} == {
        (party, leader) for party in ("alice", "bob") for leader in ("alice", "bob")
    }


def test_50_all_main_mdp_results_are_within_bound_and_have_evidence() -> None:
    data = rows(RAW / "e6_protocol_fairness_mdp_v6.csv")
    for row in data:
        assert float(row["pmax_fairness_violation"]) == 0.0
        assert float(row["pmax_terminal_fairness_violation"]) == 0.0
        assert int(row["maximum_information_lead_from_reachable_states"]) <= int(row["lambda_bits"])
        assert int(row["choices"]) > 0 and int(row["reachable_states"]) > 0 and int(row["transitions"]) > 0
        assert row["strategy_file"] and row["shortest_abort_prefix_file"] and row["model_file_sha256"]


def test_51_ell_parameter_is_consistent_in_every_reference_resource_row() -> None:
    data = rows(RAW / "e7_reference_implementation_resources_v6.csv")
    assert {int(row["ell_per_sequence"]) for row in data} == {4, 155}
    for row in data:
        n_blocks, ell = int(row["n_blocks"]), int(row["ell_per_sequence"])
        assert int(row["prepared_qubits_configured"]) == 6 * n_blocks + 2 * ell
        assert int(row["transmitted_qubits_configured"]) == 2 * n_blocks + 2 * ell
        assert int(row["local_storage_qubits_configured"]) == 4 * n_blocks
        assert int(row["protocol_total_measurements"]) == 6 * n_blocks + 2 * ell


def test_52_complete_D_decoy_has_all_six_typed_length_delimited_lists() -> None:
    disclosure = build_complete_decoy_disclosure(52, 4, 155)
    assert complete_decoy_disclosure_is_valid(disclosure, 4, 155)
    assert set(disclosure) == {"positions_S3", "bases_S3", "states_S3", "positions_S4", "bases_S4", "states_S4"}
    assert all(item["length"] == len(item["values"]) == 155 for item in disclosure.values())


@pytest.mark.parametrize("field", ["positions_S3", "bases_S3", "states_S3", "positions_S4", "bases_S4", "states_S4"])
def test_53_mutating_any_complete_decoy_field_breaks_tau_dist(field: str) -> None:
    state, raw = decoy_fixture()
    parsed = parse_record(raw)
    changed = copy.deepcopy(parsed["payload"]["D_decoy"])
    if field.startswith("positions_"):
        values = changed[field]["values"]
        replacement = next(value for value in range(5) if value not in values)
        values[0] = replacement
        values.sort()
    elif field.startswith("bases_"):
        changed[field]["values"][0] = "X" if changed[field]["values"][0] == "Z" else "Z"
    else:
        changed[field]["values"][0] ^= 1
    parsed["payload"]["D_decoy"] = changed
    parsed["record_mac"] = hmac.new(E5_KEYS.mac_a, record_auth_material(parsed["hdr"], parsed["payload"]), hashlib.sha256).hexdigest()
    forged = json.dumps(parsed, sort_keys=True, separators=(",", ":")).encode()
    assert validate_record(forged, state, E5_KEYS).reason == "tau_dist_invalid"


def test_54_decoy_metrics_are_measured_from_the_same_wire_record() -> None:
    result = IAQDReferenceExecutor(54, decoys_per_sequence=155).run_basic("00", "11")
    decoy_raw = next(raw for raw in result.wire_records if parse_record(raw)["hdr"]["type"] == "DECOY_DISCLOSURE")
    disclosure = parse_record(decoy_raw)["payload"]["D_decoy"]
    expected_payload = len(json.dumps(disclosure, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    assert result.metrics["actual_decoy_payload_bytes"] == expected_payload
    assert result.metrics["actual_decoy_record_bytes"] == len(decoy_raw)
    assert result.metrics["actual_total_wire_bytes"] == sum(map(len, result.wire_records))


def test_55_qiskit_inventory_separates_preparation_and_full_protocol_circuits() -> None:
    data = rows(RAW / "e7_qiskit_preparation_and_full_circuit_resources_v6.csv")
    assert len(data) == 2
    for row in data:
        assert float(row["preparation_fidelity"]) >= 1 - 1e-12
        assert int(row["full_single_block_single_qubit_gates_total"]) > int(row["preparation_single_qubit_gates"])
        assert int(row["full_single_block_two_qubit_gates_total"]) == int(row["preparation_two_qubit_gates"]) + 3
        assert int(row["full_single_block_measurements"]) == 6
        assert row["basis_gates"] and row["transpiler_seed"] == "20260830"


def test_56_paper_reference_and_external_resource_classes_are_separate() -> None:
    paper = rows(RAW / "e7_paper_analytic_quantum_resources_v6.csv")
    reference = rows(RAW / "e7_reference_implementation_resources_v6.csv")
    external = rows(RAW / "e7_external_ke_qkd_scope_v6.csv")
    assert paper and all(row["resource_class"] == "paper_analytic" and row["actual_wire_bytes"] == "not_applicable_paper_analytic" for row in paper)
    assert reference and all(row["resource_class"] == "reference_implementation_actual" for row in reference)
    assert external and all(row["status"] == "not_instantiated_external_primitive" and row["internal_round_count"] == "not_measured" for row in external)


def test_57_authentication_formulas_and_reference_counts_are_separate() -> None:
    data = rows(RAW / "e7_paper_vs_reference_authentication_v6.csv")
    for row in data:
        if row["mode"] == "basic":
            assert int(row["paper_logical_auth_operation_count"]) == 12
        else:
            r = int(row["r"])
            assert int(row["paper_logical_auth_operation_count"]) == 16 + 4 * r
            assert int(row["paper_commitment_generation_or_verification_count"]) == 4 * r
        assert int(row["reference_actual_auth_operation_count"]) >= int(row["paper_logical_auth_operation_count"])


def test_58_rounds_records_and_external_ke_are_not_conflated() -> None:
    data = rows(RAW / "e7_reference_implementation_resources_v6.csv")
    assert all(row["KE_internal_round_count"] == "not_measured" for row in data)
    assert all(int(row["record_count"]) != int(row["upper_layer_round_count"]) for row in data)
    assert all(int(row["upper_layer_round_count"]) == (5 if row["mode"] == "basic" else 6 + math.ceil(2 * int(row["n_blocks"]) / int(row["lambda_bits"]))) for row in data)


def test_59_performance_matrix_has_30_seeds_both_ell_values_and_success() -> None:
    data = rows(RAW / "e7_reference_performance_by_seed_v6.csv")
    grouped: dict[tuple[str, str, str, str], set[str]] = {}
    for row in data:
        key = (row["mode"], row["n_blocks"], row["lambda_bits"], row["ell_per_sequence"])
        grouped.setdefault(key, set()).add(row["seed"])
        assert row["terminal_state"] == "COMPLETED" and row["honest_success"] == "True"
    assert grouped and all(len(values) == 30 for values in grouped.values())
    assert {key[3] for key in grouped} == {"4", "155"}


def test_60_clean_reproduction_is_required_and_final_result_is_checked_when_present() -> None:
    verifier = EXP / "verify_clean_reproduction_v6.py"
    implementation = EXP / "src" / "verify_clean_reproduction_v6.py"
    assert verifier.exists() and implementation.exists()
    assert "TemporaryDirectory" in implementation.read_text(encoding="utf-8")
    result_path = EXP / "outputs" / "logs" / "clean_reproduction_v6.json"
    if result_path.exists():
        value = json.loads(result_path.read_text(encoding="utf-8"))
        assert value["status"] == "PASS" and value["all_deterministic_hashes_match"] is True
