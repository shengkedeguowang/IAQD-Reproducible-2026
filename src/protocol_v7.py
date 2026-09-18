"""Executable IAQD-REL-CAND-1 reference state machine.

This module implements the reviewed record/state interfaces without modifying
the preserved V6 executor.  It is a small reference implementation, not a
claim that the external KE, QDist assumption, MAC/PRF, or commitment has been
cryptographically instantiated for deployment.

Important design boundaries:

* Alice and Bob own separate :class:`PartyState` objects and separate key
  containers.  Public context moves only through explicit send/deliver/verify
  events.
* Bob's detection counts remain in Bob's private local dictionary.  Alice can
  learn only the authenticated QPASS decision carried by Bob's first I7
  record.
* A post-KE record has exactly one authentication mechanism.  CONFIRM uses
  the confirmation PRF and is not wrapped in a generic record MAC.
* The release counter is the local, event-based invariant S_H - V_H <= lambda;
  it is not message-knowledge fairness or effective-output fairness.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import math
import secrets
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from protocol_v6 import (  # Preserved and regression-tested quantum algebra.
    assert_quantum_block_semantics,
    build_complete_decoy_disclosure,
    complete_decoy_disclosure_is_valid,
    reference_quantum_block,
    xor_bits,
)


SPEC_ID = "IAQD-REL-CAND-1"
EXECUTOR_VERSION = "IAQD-EXEC-V7"
CANON_ID = "CE1"
ID_A = "Alice"
ID_B = "Bob"
BASIC = "BASIC"
RELEASE = "RELEASE"

HEADER_FIELDS = {
    "spec_id",
    "canon_id",
    "sid",
    "sender",
    "receiver",
    "direction",
    "type",
    "seq",
    "mode",
    "params_hash",
    "qctx",
}
AUTH_FIELDS = {"kind", "key_use", "tag"}
RECORD_FIELDS = {"hdr", "payload", "auth"}

PARAM_FIELDS = {
    "spec_id",
    "canon_id",
    "mode",
    "n",
    "message_bits",
    "ell",
    "lambda",
    "r",
    "release_lengths",
    "qber_threshold_num",
    "qber_threshold_den",
    "cluster_id",
    "pauli_id",
    "bell_id",
    "decoy_id",
    "hash_id",
    "kdf_id",
    "mac_id",
    "prf_id",
    "commitment_id",
    "otp_encoding_id",
    "mac_tag_bits",
    "conf_tag_bits",
    "salt_bits",
    "nonce_bits",
    "key_bits",
}

RECORD_PAYLOAD_FIELDS = {
    "QRECV_ACK": {"batch", "L3", "L4", "recv_complete"},
    "DECOY_INFO": {"batch", "h_ack", "decoy_count", "D_decoy"},
    "DIALOGUE_B_QPASS": {"decision", "ciphertext_bits", "ciphertext"},
    "DIALOGUE_A": {"ciphertext_bits", "ciphertext"},
    "COMMIT_SET": {"message_bits", "lambda", "r", "ciphertext_hash", "commitments"},
    "OPEN": {"j", "u_j", "chunk", "salt", "commit_set_hash"},
    "CONFIRM": {"message_bits", "confirm_label"},
}


class ProtocolViolation(RuntimeError):
    """A local sending action was attempted before its reviewed gate opened."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _u32(value: int) -> bytes:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**32:
        raise ValueError("CE1 length/count must be a uint32")
    return value.to_bytes(4, "big")


def _int_payload(value: int) -> bytes:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError("CE1 integers are non-negative integers")
    return value.to_bytes(max(1, (value.bit_length() + 7) // 8), "big")


def ce1_encode(value: Any) -> bytes:
    """Injective typed, length-delimited CE1 encoding.

    Mapping keys are UTF-8 strings encoded in lexicographic byte order.  This
    is the fixed CE1 mapping order and never depends on insertion order.
    Floats, tuples, sets, unknown Python objects, and negative integers are
    rejected.
    """

    if value is None:
        return b"N" + _u32(0)
    if isinstance(value, bool):
        return b"T" + _u32(1) + (b"1" if value else b"0")
    if isinstance(value, int):
        payload = _int_payload(value)
        return b"I" + _u32(len(payload)) + payload
    if isinstance(value, bytes):
        return b"B" + _u32(len(value)) + value
    if isinstance(value, str):
        payload = value.encode("utf-8")
        return b"S" + _u32(len(payload)) + payload
    if isinstance(value, list):
        body = b"".join(ce1_encode(item) for item in value)
        return b"L" + _u32(len(value)) + _u32(len(body)) + body
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("CE1 mapping keys must be strings")
        keys = sorted(value, key=lambda item: item.encode("utf-8"))
        body = b"".join(ce1_encode(key) + ce1_encode(value[key]) for key in keys)
        return b"D" + _u32(len(keys)) + _u32(len(body)) + body
    raise TypeError(f"unsupported CE1 value: {type(value)!r}")


def _read_u32(raw: bytes, offset: int) -> tuple[int, int]:
    if offset + 4 > len(raw):
        raise ValueError("truncated CE1 uint32")
    return int.from_bytes(raw[offset : offset + 4], "big"), offset + 4


def _ce1_decode_one(raw: bytes, offset: int) -> tuple[Any, int]:
    if offset >= len(raw):
        raise ValueError("truncated CE1 object")
    kind = raw[offset : offset + 1]
    offset += 1
    if kind in {b"N", b"T", b"I", b"B", b"S"}:
        length, offset = _read_u32(raw, offset)
        end = offset + length
        if end > len(raw):
            raise ValueError("truncated CE1 scalar")
        payload = raw[offset:end]
        if kind == b"N":
            if length != 0:
                raise ValueError("non-canonical CE1 null")
            return None, end
        if kind == b"T":
            if payload not in {b"0", b"1"}:
                raise ValueError("non-canonical CE1 boolean")
            return payload == b"1", end
        if kind == b"I":
            if not payload or (len(payload) > 1 and payload[0] == 0):
                raise ValueError("non-canonical CE1 integer")
            return int.from_bytes(payload, "big"), end
        if kind == b"B":
            return payload, end
        try:
            return payload.decode("utf-8"), end
        except UnicodeDecodeError as error:
            raise ValueError("invalid CE1 UTF-8") from error
    if kind not in {b"L", b"D"}:
        raise ValueError("unknown CE1 type tag")
    count, offset = _read_u32(raw, offset)
    body_length, offset = _read_u32(raw, offset)
    body_end = offset + body_length
    if body_end > len(raw):
        raise ValueError("truncated CE1 container")
    if kind == b"L":
        result: list[Any] = []
        for _ in range(count):
            item, offset = _ce1_decode_one(raw, offset)
            result.append(item)
        if offset != body_end:
            raise ValueError("CE1 list count/length mismatch")
        return result, body_end
    result_dict: dict[str, Any] = {}
    previous: bytes | None = None
    for _ in range(count):
        key, offset = _ce1_decode_one(raw, offset)
        if not isinstance(key, str):
            raise ValueError("CE1 map key is not a string")
        key_bytes = key.encode("utf-8")
        if previous is not None and key_bytes <= previous:
            raise ValueError("CE1 map keys are duplicated or out of order")
        previous = key_bytes
        item, offset = _ce1_decode_one(raw, offset)
        result_dict[key] = item
    if offset != body_end:
        raise ValueError("CE1 map count/length mismatch")
    return result_dict, body_end


def ce1_decode(raw: bytes) -> Any:
    if not isinstance(raw, bytes):
        raise TypeError("CE1 decoder requires bytes")
    value, offset = _ce1_decode_one(raw, 0)
    if offset != len(raw):
        raise ValueError("trailing CE1 bytes")
    if ce1_encode(value) != raw:
        raise ValueError("non-canonical CE1 encoding")
    return value


def hash_ce1(value: Any) -> str:
    return hashlib.sha256(ce1_encode(value)).hexdigest()


def _require_exact_fields(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{label}_schema_invalid")
    return value


def _require_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label}_invalid")
    return value


def _require_hex(value: Any, byte_length: int, label: str) -> str:
    if not isinstance(value, str) or len(value) != 2 * byte_length:
        raise ValueError(f"{label}_invalid")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{label}_invalid") from error
    return value.lower()


def _require_bits(value: Any, length: int, label: str) -> str:
    if not isinstance(value, str) or len(value) != length or any(bit not in "01" for bit in value):
        raise ValueError(f"{label}_invalid")
    return value


def make_params(
    *,
    mode: str,
    n: int,
    ell: int,
    lambda_bits: int | None,
    qber_threshold_num: int,
    qber_threshold_den: int,
    conf_tag_bits: int = 128,
) -> dict[str, Any]:
    if mode not in {BASIC, RELEASE}:
        raise ValueError("unsupported_mode")
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError("n_must_be_positive")
    if isinstance(ell, bool) or not isinstance(ell, int) or ell <= 0:
        raise ValueError("ell_must_be_positive")
    if (
        isinstance(qber_threshold_num, bool)
        or isinstance(qber_threshold_den, bool)
        or not isinstance(qber_threshold_num, int)
        or not isinstance(qber_threshold_den, int)
        or qber_threshold_den <= 0
        or not 0 <= qber_threshold_num <= qber_threshold_den
    ):
        raise ValueError("qber_threshold_invalid")
    if conf_tag_bits <= 0 or conf_tag_bits > 256 or conf_tag_bits % 8:
        raise ValueError("conf_tag_bits_must_be_8_to_256_and_byte_aligned")
    message_bits = 2 * n
    if mode == BASIC:
        if lambda_bits is not None:
            raise ValueError("basic_lambda_must_be_null")
        release_lengths: list[int] = []
        rounds = 0
    else:
        if isinstance(lambda_bits, bool) or not isinstance(lambda_bits, int) or not 1 <= lambda_bits <= message_bits:
            raise ValueError("release_lambda_out_of_range")
        rounds = math.ceil(message_bits / lambda_bits)
        release_lengths = [min(lambda_bits, message_bits - (j - 1) * lambda_bits) for j in range(1, rounds + 1)]
    params: dict[str, Any] = {
        "spec_id": SPEC_ID,
        "canon_id": CANON_ID,
        "mode": mode,
        "n": n,
        "message_bits": message_bits,
        "ell": ell,
        "lambda": lambda_bits,
        "r": rounds,
        "release_lengths": release_lengths,
        "qber_threshold_num": qber_threshold_num,
        "qber_threshold_den": qber_threshold_den,
        "cluster_id": "SIX-PARTICLE-CLUSTER-PHI1-PHI2",
        "pauli_id": "PAULI-2BIT",
        "bell_id": "BELL-PAIR-MEASUREMENT",
        "decoy_id": "BB84-TWO-SEQUENCE",
        "hash_id": "SHA-256",
        "kdf_id": "HKDF-LIKE-HMAC-SHA256-REFERENCE",
        "mac_id": "HMAC-SHA256-REFERENCE",
        "prf_id": "HMAC-SHA256-TRUNC-REFERENCE",
        "commitment_id": "SHA3-256-SALTED-REFERENCE",
        "otp_encoding_id": "BITSTRING-XOR",
        "mac_tag_bits": 256,
        "conf_tag_bits": conf_tag_bits,
        "salt_bits": 128,
        "nonce_bits": 128,
        "key_bits": 256,
    }
    validate_params(params)
    return params


def validate_params(params: Any) -> dict[str, Any]:
    params = _require_exact_fields(params, PARAM_FIELDS, "params")
    if params["spec_id"] != SPEC_ID or params["canon_id"] != CANON_ID:
        raise ValueError("params_version_invalid")
    if params["mode"] not in {BASIC, RELEASE}:
        raise ValueError("params_mode_invalid")
    n = _require_nonnegative_int(params["n"], "n")
    message_bits = _require_nonnegative_int(params["message_bits"], "message_bits")
    ell = _require_nonnegative_int(params["ell"], "ell")
    if n <= 0 or ell <= 0 or message_bits != 2 * n:
        raise ValueError("params_length_relation_invalid")
    numerator = _require_nonnegative_int(params["qber_threshold_num"], "qber_threshold_num")
    denominator = _require_nonnegative_int(params["qber_threshold_den"], "qber_threshold_den")
    if denominator <= 0 or numerator > denominator:
        raise ValueError("qber_threshold_invalid")
    for key in ("mac_tag_bits", "conf_tag_bits", "salt_bits", "nonce_bits", "key_bits"):
        bits = _require_nonnegative_int(params[key], key)
        if bits <= 0 or bits % 8:
            raise ValueError(f"{key}_invalid")
    if params["mac_tag_bits"] != 256 or params["conf_tag_bits"] > 256 or params["salt_bits"] < 128:
        raise ValueError("crypto_length_unsupported")
    required_identifiers = {
        "cluster_id": "SIX-PARTICLE-CLUSTER-PHI1-PHI2",
        "pauli_id": "PAULI-2BIT",
        "bell_id": "BELL-PAIR-MEASUREMENT",
        "decoy_id": "BB84-TWO-SEQUENCE",
        "hash_id": "SHA-256",
        "kdf_id": "HKDF-LIKE-HMAC-SHA256-REFERENCE",
        "mac_id": "HMAC-SHA256-REFERENCE",
        "prf_id": "HMAC-SHA256-TRUNC-REFERENCE",
        "commitment_id": "SHA3-256-SALTED-REFERENCE",
        "otp_encoding_id": "BITSTRING-XOR",
    }
    if any(params[name] != value for name, value in required_identifiers.items()):
        raise ValueError("algorithm_identifier_unsupported")
    release_lengths = params["release_lengths"]
    if not isinstance(release_lengths, list):
        raise ValueError("release_lengths_invalid")
    if params["mode"] == BASIC:
        if params["lambda"] is not None or params["r"] != 0 or release_lengths:
            raise ValueError("basic_release_parameters_invalid")
    else:
        lam = _require_nonnegative_int(params["lambda"], "lambda")
        rounds = _require_nonnegative_int(params["r"], "r")
        if not 1 <= lam <= message_bits or rounds != math.ceil(message_bits / lam):
            raise ValueError("release_parameters_invalid")
        expected = [min(lam, message_bits - (j - 1) * lam) for j in range(1, rounds + 1)]
        if release_lengths != expected:
            raise ValueError("release_lengths_invalid")
    return params


def compute_sid(params_hash: str, nonce_a: bytes, nonce_b: bytes) -> str:
    return hash_ce1(["IAQD-SID", SPEC_ID, params_hash, ID_A, ID_B, nonce_a, nonce_b])


def compute_tpre(sid: str, params_hash: str) -> str:
    return hash_ce1(["IAQD-QPRE", SPEC_ID, sid, params_hash, 0])


def compute_final_tq(
    *, sid: str, params_hash: str, length_s3: int, length_s4: int, h_ack: str, h_dist: str
) -> str:
    """Final public quantum transcript.  Numeric QBER is intentionally absent."""

    return hash_ce1(
        ["IAQD-QFINAL", SPEC_ID, sid, params_hash, 0, length_s3, length_s4, h_ack, h_dist, "QPASS"]
    )


def direction_for(sender: str) -> str:
    if sender == ID_A:
        return "A2B"
    if sender == ID_B:
        return "B2A"
    raise ValueError("unknown_sender")


def peer_for(identity: str) -> str:
    if identity == ID_A:
        return ID_B
    if identity == ID_B:
        return ID_A
    raise ValueError("unknown_identity")


@dataclass
class LocalKeys:
    ack: bytes
    dist: bytes
    mac_a: bytes
    mac_b: bytes
    conf_a: bytes
    conf_b: bytes
    otp_a: bytes
    otp_b: bytes
    key_handle: str
    counters: dict[str, int] = field(default_factory=dict)

    def count(self, name: str) -> None:
        self.counters[name] = self.counters.get(name, 0) + 1

    def mac_for(self, sender: str) -> bytes:
        return self.mac_a if sender == ID_A else self.mac_b

    def conf_for(self, sender: str) -> bytes:
        return self.conf_a if sender == ID_A else self.conf_b

    def otp_for(self, sender: str) -> bytes:
        return self.otp_a if sender == ID_A else self.otp_b


def _expand(master: bytes, sid: str, label: str, length: int) -> bytes:
    output = bytearray()
    counter = 1
    while len(output) < length:
        output.extend(hmac.new(master, ce1_encode(["IAQD-KDF", sid, label, counter]), hashlib.sha256).digest())
        counter += 1
    return bytes(output[:length])


def derive_local_keys(master: bytes, sid: str, message_bits: int) -> LocalKeys:
    otp_bytes = max(1, math.ceil(message_bits / 8))
    return LocalKeys(
        ack=_expand(master, sid, "K_ack", 32),
        dist=_expand(master, sid, "K_dist", 32),
        mac_a=_expand(master, sid, "K_A_mac", 32),
        mac_b=_expand(master, sid, "K_B_mac", 32),
        conf_a=_expand(master, sid, "K_A_conf", 32),
        conf_b=_expand(master, sid, "K_B_conf", 32),
        otp_a=_expand(master, sid, "K_A_otp", otp_bytes),
        otp_b=_expand(master, sid, "K_B_otp", otp_bytes),
        key_handle=secrets.token_hex(16),
    )


def _key_bits(key: bytes, length: int) -> str:
    return f"{int.from_bytes(key, 'big'):0{8 * len(key)}b}"[-length:]


@dataclass
class PartyState:
    identity: str
    peer_identity: str
    sid: str
    params: dict[str, Any]
    params_hash: str
    t_pre: str
    keys: LocalKeys = field(repr=False)
    ke_context: str
    active: bool = True
    terminated: bool = False
    terminal_reason: str | None = None
    completed: bool = False
    phase: str = "KE_ACCEPTED"
    next_send_seq: int = 0
    next_recv_seq: int = 0
    sent_tokens: set[str] = field(default_factory=set)
    accepted_tokens: set[str] = field(default_factory=set)
    accepted_record_hashes: set[str] = field(default_factory=set)
    public_hashes: dict[str, str] = field(default_factory=dict)
    ack_lengths: tuple[int, int] | None = None
    quantum_sent: bool = False
    quantum_received: bool = False
    qrecv_sent: bool = False
    qrecv_accepted: bool = False
    decoy_sent: bool = False
    decoy_accepted: bool = False
    detection_passed: bool = False
    qpass_authenticated: bool = False
    t_q: str | None = None
    dialogue_b_sent: bool = False
    dialogue_a_sent: bool = False
    ciphertexts: dict[str, str] = field(default_factory=dict)
    quantum_view: dict[str, str] = field(default_factory=dict, repr=False)
    recovered_peer_message: str | None = field(default=None, repr=False)
    commit_set_sent: bool = False
    peer_commit_set: dict[int, dict[str, Any]] = field(default_factory=dict)
    own_open_sent: set[int] = field(default_factory=set)
    peer_open_accepted: set[int] = field(default_factory=set)
    peer_mask_bits: str = field(default="", repr=False)
    S: int = 0
    V: int = 0
    max_gap: int = 0
    confirm_sent: bool = False
    peer_confirm_accepted: bool = False
    private_local: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def mode(self) -> str:
        return str(self.params["mode"])

    @property
    def message_bits(self) -> int:
        return int(self.params["message_bits"])

    @property
    def outgoing_direction(self) -> str:
        return direction_for(self.identity)

    @property
    def incoming_direction(self) -> str:
        return direction_for(self.peer_identity)

    def own_message(self) -> str:
        return str(self.private_local["own_message"])

    def own_mask(self) -> str:
        return str(self.private_local["own_mask"])

    def snapshot(self) -> dict[str, Any]:
        """Audit snapshot excluding keys, messages, masks, salts and QBER counts."""

        return {
            "identity": self.identity,
            "sid": self.sid,
            "mode": self.mode,
            "params_hash": self.params_hash,
            "active": self.active,
            "terminated": self.terminated,
            "terminal_reason": self.terminal_reason,
            "completed": self.completed,
            "phase": self.phase,
            "next_send_seq": self.next_send_seq,
            "next_recv_seq": self.next_recv_seq,
            "sent_tokens": sorted(self.sent_tokens),
            "accepted_tokens": sorted(self.accepted_tokens),
            "public_hashes": dict(sorted(self.public_hashes.items())),
            "ack_lengths": None if self.ack_lengths is None else list(self.ack_lengths),
            "quantum_sent": self.quantum_sent,
            "quantum_received": self.quantum_received,
            "qrecv_sent": self.qrecv_sent,
            "qrecv_accepted": self.qrecv_accepted,
            "decoy_sent": self.decoy_sent,
            "decoy_accepted": self.decoy_accepted,
            "detection_passed": self.detection_passed,
            "qpass_authenticated": self.qpass_authenticated,
            "t_q": self.t_q,
            "ciphertext_roles": sorted(self.ciphertexts),
            "quantum_view_fields": sorted(self.quantum_view),
            "recovered_peer_message_present": self.recovered_peer_message is not None,
            "commit_set_sent": self.commit_set_sent,
            "peer_commit_indices": sorted(self.peer_commit_set),
            "own_open_sent": sorted(self.own_open_sent),
            "peer_open_accepted": sorted(self.peer_open_accepted),
            "S": self.S,
            "V": self.V,
            "current_gap": self.S - self.V,
            "max_gap": self.max_gap,
            "confirm_sent": self.confirm_sent,
            "peer_confirm_accepted": self.peer_confirm_accepted,
            "private_fields_present": sorted(self.private_local),
            "key_handle": self.keys.key_handle,
            "auth_counters": dict(sorted(self.keys.counters.items())),
        }


@dataclass
class ValidationResult:
    accepted: bool
    reason: str
    record_hash: str | None
    trace: list[str]


@dataclass
class ExecutionResult:
    mode: str
    sid: str
    success: bool
    terminal_state: str
    alice_state: dict[str, Any]
    bob_state: dict[str, Any]
    recovered_by_alice: str | None
    recovered_by_bob: str | None
    event_log: list[dict[str, Any]]
    wire_records: list[bytes]
    metrics: dict[str, Any]


def parse_record(raw_record: bytes) -> dict[str, Any]:
    record = ce1_decode(raw_record)
    _require_exact_fields(record, RECORD_FIELDS, "record")
    _require_exact_fields(record["hdr"], HEADER_FIELDS, "header")
    if not isinstance(record["payload"], dict):
        raise ValueError("payload_schema_invalid")
    _require_exact_fields(record["auth"], AUTH_FIELDS, "auth")
    return record


def _mac_tag(domain: str, key: bytes, hdr: dict[str, Any], payload: dict[str, Any]) -> str:
    return hmac.new(key, ce1_encode([domain, hdr, payload]), hashlib.sha256).hexdigest()


def _auth_spec(record_type: str, sender: str) -> tuple[str, str, str]:
    if record_type == "QRECV_ACK":
        return "MAC", "K_ack", "IAQD-ACK"
    if record_type == "DECOY_INFO":
        return "MAC", "K_dist", "IAQD-DECOY"
    if record_type == "CONFIRM":
        return "PRF", f"K_{'A' if sender == ID_A else 'B'}_conf", "IAQD-I8"
    if record_type in {"DIALOGUE_B_QPASS", "DIALOGUE_A", "COMMIT_SET", "OPEN"}:
        return "MAC", f"K_{'A' if sender == ID_A else 'B'}_mac", "IAQD-RECORD"
    raise ValueError("unsupported_record_type")


def _confirm_context(state: PartyState, hdr: dict[str, Any], payload: dict[str, Any], message: str) -> list[Any]:
    if set(state.ciphertexts) != {"A", "B"} or state.t_q is None:
        raise ValueError("confirmation_context_unavailable")
    return [
        "IAQD-I8",
        SPEC_ID,
        CANON_ID,
        state.sid,
        state.mode,
        state.params_hash,
        hdr["sender"],
        hdr["receiver"],
        hdr["direction"],
        "CONFIRM",
        hdr["seq"],
        state.message_bits,
        message,
        state.ciphertexts["A"],
        state.ciphertexts["B"],
        state.t_q,
        payload["confirm_label"],
    ]


def _confirm_tag(state: PartyState, hdr: dict[str, Any], payload: dict[str, Any], message: str) -> str:
    tag_bytes = int(state.params["conf_tag_bits"]) // 8
    full = hmac.new(state.keys.conf_for(str(hdr["sender"])), ce1_encode(_confirm_context(state, hdr, payload, message)), hashlib.sha256).digest()
    return full[:tag_bytes].hex()


def _build_record(state: PartyState, record_type: str, payload: dict[str, Any], qctx: str, seq: int) -> bytes:
    sender = state.identity
    hdr = {
        "spec_id": SPEC_ID,
        "canon_id": CANON_ID,
        "sid": state.sid,
        "sender": sender,
        "receiver": state.peer_identity,
        "direction": state.outgoing_direction,
        "type": record_type,
        "seq": seq,
        "mode": state.mode,
        "params_hash": state.params_hash,
        "qctx": qctx,
    }
    kind, key_use, domain = _auth_spec(record_type, sender)
    if kind == "PRF":
        tag = _confirm_tag(state, hdr, payload, state.own_message())
        state.keys.count("confirm_prf_generate")
    else:
        if key_use == "K_ack":
            key = state.keys.ack
            state.keys.count("ack_mac_generate")
        elif key_use == "K_dist":
            key = state.keys.dist
            state.keys.count("decoy_mac_generate")
        else:
            key = state.keys.mac_for(sender)
            state.keys.count(f"{record_type.lower()}_mac_generate")
        tag = _mac_tag(domain, key, hdr, payload)
    return ce1_encode({"hdr": hdr, "payload": payload, "auth": {"kind": kind, "key_use": key_use, "tag": tag}})


def record_with_changes(raw_record: bytes, changes: Mapping[tuple[str, ...], Any]) -> bytes:
    """Syntax-only test utility: mutate decoded fields without recomputing auth."""

    record = parse_record(raw_record)
    for path, value in changes.items():
        if not path:
            raise ValueError("empty_change_path")
        cursor: Any = record
        for key in path[:-1]:
            cursor = cursor[key]
        cursor[path[-1]] = value
    return ce1_encode(record)


def encode_record_object(record: dict[str, Any]) -> bytes:
    """Encode a caller-supplied record object; validation remains authoritative."""

    return ce1_encode(record)


def _expected_type(state: PartyState) -> str:
    seq = state.next_recv_seq
    if state.identity == ID_A:
        if seq == 0:
            return "QRECV_ACK"
        if seq == 1:
            return "DIALOGUE_B_QPASS"
    else:
        if seq == 0:
            return "DECOY_INFO"
        if seq == 1:
            return "DIALOGUE_A"
    if state.mode == BASIC:
        if seq == 2:
            return "CONFIRM"
        return "NO_FURTHER_RECORD"
    rounds = int(state.params["r"])
    if seq == 2:
        return "COMMIT_SET"
    if 3 <= seq <= rounds + 2:
        return "OPEN"
    if seq == rounds + 3:
        return "CONFIRM"
    return "NO_FURTHER_RECORD"


def _reject(state: PartyState, reason: str, trace: list[str]) -> ValidationResult:
    state.active = False
    state.terminated = True
    state.terminal_reason = reason
    state.phase = "TERMINATED"
    return ValidationResult(False, reason, None, trace + [f"reject:{reason}"])


def _try_recover(state: PartyState) -> None:
    peer_role = "B" if state.identity == ID_A else "A"
    if peer_role not in state.ciphertexts or "shared_M_peer" not in state.quantum_view:
        return
    if state.mode == RELEASE and len(state.peer_mask_bits) != state.message_bits:
        return
    peer_identity = state.peer_identity
    otp = _key_bits(state.keys.otp_for(peer_identity), state.message_bits)
    values = [state.ciphertexts[peer_role], otp]
    if state.mode == RELEASE:
        values.append(state.peer_mask_bits)
    tilde_peer = xor_bits(*values)
    state.recovered_peer_message = xor_bits(state.quantum_view["shared_M_peer"], tilde_peer)
    state.phase = "PEER_MESSAGE_RECOVERED"


def _validate_commit_set(state: PartyState, hdr: dict[str, Any], payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    if state.mode != RELEASE or set(state.ciphertexts) != {"A", "B"} or state.t_q is None:
        raise ValueError("commit_set_stage_invalid")
    if payload["message_bits"] != state.message_bits or payload["lambda"] != state.params["lambda"] or payload["r"] != state.params["r"]:
        raise ValueError("commit_set_parameters_invalid")
    peer_role = "A" if hdr["sender"] == ID_A else "B"
    if payload["ciphertext_hash"] != hash_ce1(state.ciphertexts[peer_role]):
        raise ValueError("commit_set_ciphertext_hash_invalid")
    entries = payload["commitments"]
    if not isinstance(entries, list) or len(entries) != state.params["r"]:
        raise ValueError("commit_set_entries_invalid")
    result: dict[int, dict[str, Any]] = {}
    for expected_j, item in enumerate(entries, 1):
        _require_exact_fields(item, {"j", "u_j", "d_j"}, "commitment")
        if item["j"] != expected_j or item["u_j"] != state.params["release_lengths"][expected_j - 1]:
            raise ValueError("commitment_index_or_length_invalid")
        _require_hex(item["d_j"], 32, "commitment_digest")
        result[expected_j] = dict(item)
    return result


def compute_commitment(
    *,
    state: PartyState,
    sender: str,
    j: int,
    u_j: int,
    ciphertext: str,
    chunk: str,
    salt: bytes,
) -> str:
    direction = direction_for(sender)
    material = [
        "IAQD-COMMIT",
        SPEC_ID,
        state.sid,
        state.params_hash,
        state.t_q,
        sender,
        direction,
        j,
        u_j,
        hash_ce1(ciphertext),
        chunk,
        salt,
    ]
    return hashlib.sha3_256(ce1_encode(material)).hexdigest()


def _candidate_tq_for_alice(state: PartyState, payload: dict[str, Any]) -> str:
    if (
        state.identity != ID_A
        or not state.qrecv_accepted
        or not state.decoy_sent
        or state.ack_lengths is None
        or "h_ack" not in state.public_hashes
        or "h_dist" not in state.public_hashes
        or payload.get("decision") != "QPASS"
    ):
        raise ValueError("qpass_context_unavailable")
    return compute_final_tq(
        sid=state.sid,
        params_hash=state.params_hash,
        length_s3=state.ack_lengths[0],
        length_s4=state.ack_lengths[1],
        h_ack=state.public_hashes["h_ack"],
        h_dist=state.public_hashes["h_dist"],
    )


def validate_record(raw_record: bytes, receiver_state: PartyState) -> ValidationResult:
    """Validate using only the received bytes and the receiver's local state."""

    state = receiver_state
    trace = ["parse_ce1"]
    if state.terminated:
        return ValidationResult(False, "session_already_terminated", None, trace + ["reject:session_already_terminated"])
    try:
        record = parse_record(raw_record)
    except (TypeError, ValueError) as error:
        return _reject(state, str(error), trace)
    hdr = record["hdr"]
    payload = record["payload"]
    auth = record["auth"]
    record_hash = hashlib.sha256(raw_record).hexdigest()

    trace.append("check_header_context")
    if hdr["spec_id"] != SPEC_ID or hdr["canon_id"] != CANON_ID:
        return _reject(state, "version_or_canon_mismatch", trace)
    if hdr["sid"] != state.sid:
        return _reject(state, "sid_mismatch", trace)
    if hdr["sender"] != state.peer_identity or hdr["receiver"] != state.identity:
        return _reject(state, "identity_mismatch", trace)
    if hdr["direction"] != state.incoming_direction:
        return _reject(state, "direction_mismatch", trace)
    if hdr["mode"] != state.mode:
        return _reject(state, "mode_mismatch", trace)
    if hdr["params_hash"] != state.params_hash:
        return _reject(state, "params_hash_mismatch", trace)
    expected_type = _expected_type(state)
    if hdr["type"] != expected_type:
        return _reject(state, "unexpected_type_or_stage", trace)
    if isinstance(hdr["seq"], bool) or not isinstance(hdr["seq"], int):
        return _reject(state, "sequence_type_invalid", trace)
    if hdr["seq"] < state.next_recv_seq:
        return _reject(state, "replayed_or_old_sequence", trace)
    if hdr["seq"] > state.next_recv_seq:
        return _reject(state, "future_sequence", trace)
    token = f"{state.sid}:{hdr['direction']}:{hdr['seq']}:{hdr['type']}"
    if token in state.accepted_tokens or record_hash in state.accepted_record_hashes:
        return _reject(state, "duplicate_record", trace)

    record_type = str(hdr["type"])
    trace.append("check_payload_schema")
    required_payload = RECORD_PAYLOAD_FIELDS.get(record_type)
    if required_payload is None or set(payload) != required_payload:
        return _reject(state, "payload_schema_invalid", trace)

    try:
        if record_type == "DIALOGUE_B_QPASS":
            expected_qctx = _candidate_tq_for_alice(state, payload)
        elif record_type in {"QRECV_ACK", "DECOY_INFO"}:
            expected_qctx = state.t_pre
        else:
            if state.t_q is None:
                raise ValueError("final_tq_unavailable")
            expected_qctx = state.t_q
    except ValueError as error:
        return _reject(state, str(error), trace)
    if hdr["qctx"] != expected_qctx:
        return _reject(state, "tq_mismatch", trace)

    # Semantic checks prepare a deferred update; no accepted-state field changes
    # before every check and the one prescribed authenticator succeeds.
    updates: dict[str, Any] = {}
    try:
        if record_type == "QRECV_ACK":
            if state.identity != ID_A or not state.quantum_sent:
                raise ValueError("qrecv_stage_invalid")
            if payload["batch"] != 0 or payload["recv_complete"] != 1:
                raise ValueError("qrecv_marker_invalid")
            expected_length = int(state.params["n"]) + int(state.params["ell"])
            if payload["L3"] != expected_length or payload["L4"] != expected_length:
                raise ValueError("qrecv_length_invalid")
            updates["ack_lengths"] = (int(payload["L3"]), int(payload["L4"]))
        elif record_type == "DECOY_INFO":
            if state.identity != ID_B or not state.qrecv_sent:
                raise ValueError("decoy_stage_invalid")
            if payload["batch"] != 0 or payload["h_ack"] != state.public_hashes.get("h_ack"):
                raise ValueError("decoy_ack_binding_invalid")
            if payload["decoy_count"] != 2 * int(state.params["ell"]):
                raise ValueError("decoy_count_invalid")
            if not complete_decoy_disclosure_is_valid(payload["D_decoy"], int(state.params["n"]), int(state.params["ell"])):
                raise ValueError("decoy_description_invalid")
        elif record_type == "DIALOGUE_B_QPASS":
            if payload["decision"] != "QPASS" or payload["ciphertext_bits"] != state.message_bits:
                raise ValueError("qpass_or_length_invalid")
            updates["peer_ciphertext"] = _require_bits(payload["ciphertext"], state.message_bits, "ciphertext")
            updates["t_q"] = expected_qctx
        elif record_type == "DIALOGUE_A":
            if state.identity != ID_B or not state.dialogue_b_sent or not state.detection_passed:
                raise ValueError("dialogue_a_stage_invalid")
            if payload["ciphertext_bits"] != state.message_bits:
                raise ValueError("dialogue_length_invalid")
            updates["peer_ciphertext"] = _require_bits(payload["ciphertext"], state.message_bits, "ciphertext")
        elif record_type == "COMMIT_SET":
            updates["peer_commit_set"] = _validate_commit_set(state, hdr, payload)
        elif record_type == "OPEN":
            if not state.peer_commit_set:
                raise ValueError("opening_without_commit_set")
            j = _require_nonnegative_int(payload["j"], "opening_index")
            if j != len(state.peer_open_accepted) + 1:
                raise ValueError("opening_out_of_order_or_duplicate")
            u_j = int(state.params["release_lengths"][j - 1]) if 1 <= j <= state.params["r"] else -1
            if payload["u_j"] != u_j:
                raise ValueError("opening_length_invalid")
            chunk = _require_bits(payload["chunk"], u_j, "opening_chunk")
            salt_hex = _require_hex(payload["salt"], int(state.params["salt_bits"]) // 8, "opening_salt")
            expected_commit_hash = state.public_hashes.get("peer_commit_set_hash")
            if payload["commit_set_hash"] != expected_commit_hash:
                raise ValueError("commit_set_hash_mismatch")
            peer_role = "A" if hdr["sender"] == ID_A else "B"
            actual_commitment = compute_commitment(
                state=state,
                sender=str(hdr["sender"]),
                j=j,
                u_j=u_j,
                ciphertext=state.ciphertexts[peer_role],
                chunk=chunk,
                salt=bytes.fromhex(salt_hex),
            )
            if not hmac.compare_digest(actual_commitment, state.peer_commit_set[j]["d_j"]):
                raise ValueError("commitment_opening_mismatch")
            updates.update({"opening_j": j, "opening_u": u_j, "opening_chunk": chunk})
        elif record_type == "CONFIRM":
            if payload["message_bits"] != state.message_bits or payload["confirm_label"] != "FULL-RECOVERY":
                raise ValueError("confirmation_payload_invalid")
            if state.recovered_peer_message is None:
                raise ValueError("confirmation_before_recovery")
            if state.mode == RELEASE and (
                state.S != state.message_bits
                or state.V != state.message_bits
                or len(state.own_open_sent) != state.params["r"]
                or len(state.peer_open_accepted) != state.params["r"]
            ):
                raise ValueError("release_confirmation_gate_closed")
    except (IndexError, TypeError, ValueError) as error:
        return _reject(state, str(error), trace)

    trace.append("verify_prescribed_authenticator")
    try:
        kind, key_use, domain = _auth_spec(record_type, str(hdr["sender"]))
        if auth["kind"] != kind or auth["key_use"] != key_use:
            raise ValueError("auth_dispatch_mismatch")
        if kind == "PRF":
            state.keys.count("confirm_prf_verify")
            trace.append("verify_auth:PRF")
            expected_tag = _confirm_tag(state, hdr, payload, str(state.recovered_peer_message))
            _require_hex(auth["tag"], int(state.params["conf_tag_bits"]) // 8, "confirmation_tag")
        else:
            trace.append(f"verify_auth:{key_use}")
            if key_use == "K_ack":
                key = state.keys.ack
                state.keys.count("ack_mac_verify")
            elif key_use == "K_dist":
                key = state.keys.dist
                state.keys.count("decoy_mac_verify")
            else:
                key = state.keys.mac_for(str(hdr["sender"]))
                state.keys.count(f"{record_type.lower()}_mac_verify")
            expected_tag = _mac_tag(domain, key, hdr, payload)
            _require_hex(auth["tag"], int(state.params["mac_tag_bits"]) // 8, "mac_tag")
        if not hmac.compare_digest(str(auth["tag"]), expected_tag):
            raise ValueError("authentication_tag_invalid")
    except (TypeError, ValueError) as error:
        return _reject(state, str(error), trace)

    # Atomic accepted-state update.
    state.accepted_tokens.add(token)
    state.accepted_record_hashes.add(record_hash)
    state.next_recv_seq += 1
    if record_type == "QRECV_ACK":
        state.qrecv_accepted = True
        state.ack_lengths = updates["ack_lengths"]
        state.public_hashes["h_ack"] = record_hash
        state.phase = "ACK_ACCEPTED"
    elif record_type == "DECOY_INFO":
        state.decoy_accepted = True
        state.public_hashes["h_dist"] = record_hash
        state.phase = "DECOY_ACCEPTED"
    elif record_type == "DIALOGUE_B_QPASS":
        state.t_q = updates["t_q"]
        state.qpass_authenticated = True
        state.ciphertexts["B"] = updates["peer_ciphertext"]
        state.phase = "BOB_DIALOGUE_ACCEPTED"
        _try_recover(state)
    elif record_type == "DIALOGUE_A":
        state.ciphertexts["A"] = updates["peer_ciphertext"]
        state.phase = "ALICE_DIALOGUE_ACCEPTED"
        _try_recover(state)
    elif record_type == "COMMIT_SET":
        state.peer_commit_set = updates["peer_commit_set"]
        state.public_hashes["peer_commit_set_hash"] = record_hash
        state.phase = "PEER_COMMIT_SET_ACCEPTED"
    elif record_type == "OPEN":
        state.peer_open_accepted.add(updates["opening_j"])
        state.peer_mask_bits += updates["opening_chunk"]
        state.V += updates["opening_u"]
        state.max_gap = max(state.max_gap, state.S - state.V)
        state.phase = "PEER_OPEN_ACCEPTED"
        _try_recover(state)
    elif record_type == "CONFIRM":
        state.peer_confirm_accepted = True
        state.phase = "PEER_CONFIRM_ACCEPTED"
        if state.confirm_sent:
            state.active = False
            state.terminated = True
            state.completed = True
            state.terminal_reason = "COMPLETED"
            state.phase = "COMPLETED"
    trace.append("accept_and_commit_state")
    return ValidationResult(True, "accepted", record_hash, trace)


class IAQDSession:
    """Two-party event scheduler around two independent local state machines."""

    def __init__(
        self,
        *,
        mode: str,
        message_a: str,
        message_b: str,
        lambda_bits: int | None = None,
        ell: int = 4,
        qber_threshold_num: int = 11,
        qber_threshold_den: int = 100,
        conf_tag_bits: int = 128,
        quantum_blocks: Sequence[dict[str, Any]] | None = None,
    ):
        if len(message_a) != len(message_b) or not message_a or len(message_a) % 2:
            raise ValueError("messages_must_have_equal_positive_even_length")
        _require_bits(message_a, len(message_a), "message_a")
        _require_bits(message_b, len(message_b), "message_b")
        n = len(message_a) // 2
        self.params = make_params(
            mode=mode,
            n=n,
            ell=ell,
            lambda_bits=lambda_bits,
            qber_threshold_num=qber_threshold_num,
            qber_threshold_den=qber_threshold_den,
            conf_tag_bits=conf_tag_bits,
        )
        self.params_hash = hash_ce1(self.params)
        self.nonce_a = secrets.token_bytes(16)
        self.nonce_b = secrets.token_bytes(16)
        self.sid = compute_sid(self.params_hash, self.nonce_a, self.nonce_b)
        self.t_pre = compute_tpre(self.sid, self.params_hash)
        init_a = {
            "type": "INIT_A",
            "spec_id": SPEC_ID,
            "canon_id": CANON_ID,
            "proposer": ID_A,
            "peer": ID_B,
            "N_A": self.nonce_a,
            "params": self.params,
        }
        init_b = {
            "type": "INIT_B",
            "spec_id": SPEC_ID,
            "canon_id": CANON_ID,
            "responder": ID_B,
            "peer": ID_A,
            "N_B": self.nonce_b,
            "params_hash": self.params_hash,
            "h_init_a": hash_ce1(init_a),
            "accept": "EXACT",
        }
        t_init = hash_ce1(["IAQD-INIT", SPEC_ID, self.sid, self.params_hash, hash_ce1(init_a), hash_ce1(init_b)])
        self.ke_context = hash_ce1([SPEC_ID, self.sid, t_init, self.params_hash, ID_A, ID_B])
        # Ideal authenticated KE fixture: equality is provided by KE, while the
        # application sees only separate local outputs and fresh key handles.
        master = secrets.token_bytes(32)
        keys_a = derive_local_keys(master, self.sid, len(message_a))
        keys_b = derive_local_keys(master, self.sid, len(message_a))
        self.alice = PartyState(
            identity=ID_A,
            peer_identity=ID_B,
            sid=self.sid,
            params=copy.deepcopy(self.params),
            params_hash=self.params_hash,
            t_pre=self.t_pre,
            keys=keys_a,
            ke_context=self.ke_context,
            private_local={"own_message": message_a},
        )
        self.bob = PartyState(
            identity=ID_B,
            peer_identity=ID_A,
            sid=self.sid,
            params=copy.deepcopy(self.params),
            params_hash=self.params_hash,
            t_pre=self.t_pre,
            keys=keys_b,
            ke_context=self.ke_context,
            private_local={"own_message": message_b},
        )
        self.event_log: list[dict[str, Any]] = []
        self.wire_records: list[bytes] = []
        self.quantum_blocks = [copy.deepcopy(block) for block in quantum_blocks] if quantum_blocks is not None else None
        self._event("Alice", "INIT_A_SENT", result="sent", object_hash=hash_ce1(init_a))
        self._event("Bob", "INIT_A_RECEIVED", result="accepted_pending_ke", object_hash=hash_ce1(init_a))
        self._event("Bob", "INIT_B_SENT", result="sent", object_hash=hash_ce1(init_b))
        self._event("Alice", "INIT_B_RECEIVED", result="accepted_pending_ke", object_hash=hash_ce1(init_b))
        self._event("Alice", "KE_ACCEPT", result="accepted", context=self.ke_context, key_handle=keys_a.key_handle)
        self._event("Bob", "KE_ACCEPT", result="accepted", context=self.ke_context, key_handle=keys_b.key_handle)

    def _party(self, identity: str) -> PartyState:
        return self.alice if identity == ID_A else self.bob

    def _event(self, actor: str, event: str, **details: Any) -> None:
        self.event_log.append({"index": len(self.event_log), "actor": actor, "event": event, "sid": self.sid, **details})

    def _require_active(self, state: PartyState) -> None:
        if not state.active or state.terminated:
            raise ProtocolViolation("local_session_not_active")

    def _record_send(self, state: PartyState, raw: bytes, record_type: str, before: dict[str, Any]) -> None:
        record_hash = hashlib.sha256(raw).hexdigest()
        self.wire_records.append(raw)
        state.sent_tokens.add(f"{state.sid}:{state.outgoing_direction}:{state.next_send_seq}:{record_type}")
        state.next_send_seq += 1
        self._event(
            state.identity,
            "RECORD_SEND",
            result="sent",
            record_type=record_type,
            direction=state.outgoing_direction,
            seq=state.next_send_seq - 1,
            record_hash=record_hash,
            state_before=before,
            state_after=state.snapshot(),
        )

    def deliver(self, raw: bytes, receiver: str) -> ValidationResult:
        state = self._party(receiver)
        before = state.snapshot()
        result = validate_record(raw, state)
        record_type = "UNPARSEABLE"
        seq: int | None = None
        try:
            parsed = parse_record(raw)
            record_type = str(parsed["hdr"]["type"])
            seq = int(parsed["hdr"]["seq"]) if isinstance(parsed["hdr"]["seq"], int) else None
        except (TypeError, ValueError):
            pass
        self._event(
            receiver,
            "RECORD_DELIVER_VERIFY",
            result="accepted" if result.accepted else "rejected",
            reason=result.reason,
            record_type=record_type,
            seq=seq,
            record_hash=hashlib.sha256(raw).hexdigest(),
            validation_trace=result.trace,
            state_before=before,
            state_after=state.snapshot(),
        )
        return result

    def inject_and_deliver(self, raw: bytes, receiver: str) -> ValidationResult:
        """Record an externally supplied wire object, then deliver it locally."""

        self.wire_records.append(raw)
        self._event("network", "RECORD_INJECT", result="delivered", record_hash=hashlib.sha256(raw).hexdigest())
        return self.deliver(raw, receiver)

    def send_quantum(self, *, length_s3: int | None = None, length_s4: int | None = None) -> bool:
        self._require_active(self.alice)
        expected = int(self.params["n"]) + int(self.params["ell"])
        l3 = expected if length_s3 is None else int(length_s3)
        l4 = expected if length_s4 is None else int(length_s4)
        before_a = self.alice.snapshot()
        self.alice.quantum_sent = True
        self.alice.phase = "WAIT_QRECV_ACK"
        self._event(
            ID_A,
            "QUANTUM_SEND",
            result="sent",
            batch=0,
            L3=l3,
            L4=l4,
            state_before=before_a,
            state_after=self.alice.snapshot(),
        )
        before_b = self.bob.snapshot()
        if l3 != expected or l4 != expected:
            self.bob.active = False
            self.bob.terminated = True
            self.bob.terminal_reason = "QUANTUM_LENGTH_INVALID"
            self.bob.phase = "TERMINATED"
            accepted = False
            reason = "quantum_length_invalid"
        else:
            self.bob.quantum_received = True
            self.bob.ack_lengths = (l3, l4)
            self.bob.phase = "QUANTUM_RECEIVED"
            accepted = True
            reason = "accepted"
        self._event(
            ID_B,
            "QUANTUM_DELIVER_LENGTH_CHECK",
            result="accepted" if accepted else "rejected",
            reason=reason,
            batch=0,
            L3=l3,
            L4=l4,
            state_before=before_b,
            state_after=self.bob.snapshot(),
        )
        return accepted

    def send_qrecv_ack(self) -> bytes:
        state = self.bob
        self._require_active(state)
        if not state.quantum_received or state.ack_lengths is None or state.next_send_seq != 0:
            raise ProtocolViolation("qrecv_send_gate_closed")
        payload = {"batch": 0, "L3": state.ack_lengths[0], "L4": state.ack_lengths[1], "recv_complete": 1}
        before = state.snapshot()
        raw = _build_record(state, "QRECV_ACK", payload, state.t_pre, 0)
        state.qrecv_sent = True
        state.public_hashes["h_ack"] = hashlib.sha256(raw).hexdigest()
        state.phase = "QRECV_ACK_SENT"
        self._record_send(state, raw, "QRECV_ACK", before)
        return raw

    def send_decoy_info(self) -> bytes:
        state = self.alice
        self._require_active(state)
        if not state.qrecv_accepted or state.next_send_seq != 0:
            raise ProtocolViolation("decoy_requires_locally_verified_qrecv")
        disclosure = build_complete_decoy_disclosure(secrets.randbits(64), int(self.params["n"]), int(self.params["ell"]))
        payload = {
            "batch": 0,
            "h_ack": state.public_hashes["h_ack"],
            "decoy_count": 2 * int(self.params["ell"]),
            "D_decoy": disclosure,
        }
        before = state.snapshot()
        raw = _build_record(state, "DECOY_INFO", payload, state.t_pre, 0)
        state.decoy_sent = True
        state.public_hashes["h_dist"] = hashlib.sha256(raw).hexdigest()
        state.phase = "DECOY_INFO_SENT"
        self._record_send(state, raw, "DECOY_INFO", before)
        return raw

    def bob_detection(self, *, errors: int = 0, trials: int | None = None) -> bool:
        state = self.bob
        self._require_active(state)
        if not state.decoy_accepted:
            raise ProtocolViolation("detection_requires_verified_decoy_info")
        actual_trials = 2 * int(self.params["ell"]) if trials is None else int(trials)
        if isinstance(errors, bool) or not isinstance(errors, int) or not 0 <= errors <= actual_trials or actual_trials <= 0:
            raise ValueError("detection_counts_invalid")
        state.private_local["detection_counts"] = (errors, actual_trials)
        passed = errors * int(self.params["qber_threshold_den"]) <= actual_trials * int(self.params["qber_threshold_num"])
        before = state.snapshot()
        state.detection_passed = passed
        if passed:
            state.phase = "DETECTION_QPASS_LOCAL"
            result = "QPASS"
        else:
            state.active = False
            state.terminated = True
            state.terminal_reason = "DETECTION_FAILED"
            state.phase = "TERMINATED"
            result = "QFAIL_LOCAL"
        self._event(
            ID_B,
            "LOCAL_DETECTION_DECISION",
            result=result,
            # Counts remain in Bob.private_local and are intentionally absent.
            state_before=before,
            state_after=state.snapshot(),
        )
        return passed

    def perform_quantum_core(self) -> None:
        if not self.bob.detection_passed:
            raise ProtocolViolation("quantum_core_requires_bob_qpass")
        message_a = self.alice.own_message()
        message_b = self.bob.own_message()
        n = int(self.params["n"])
        blocks = self.quantum_blocks
        if blocks is None:
            blocks = [
                reference_quantum_block("phi1" if index % 2 == 0 else "phi2", message_a[2 * index : 2 * index + 2], message_b[2 * index : 2 * index + 2])
                for index in range(n)
            ]
        if len(blocks) != n:
            raise ValueError("quantum_block_count_invalid")
        checked = [
            assert_quantum_block_semantics(block, message_a[2 * index : 2 * index + 2], message_b[2 * index : 2 * index + 2])
            for index, block in enumerate(blocks)
        ]
        m_a = "".join(item["M_A"] for item in checked)
        m_b = "".join(item["M_B"] for item in checked)
        tilde_a = "".join(item["tilde_M_A"] for item in checked)
        tilde_b = "".join(item["tilde_M_B"] for item in checked)
        self.alice.quantum_view = {"shared_M_own": m_a, "shared_M_peer": m_b, "tilde_own": tilde_a}
        self.bob.quantum_view = {"shared_M_own": m_b, "shared_M_peer": m_a, "tilde_own": tilde_b}
        self.alice.phase = "LOCAL_QUANTUM_CORE_READY"
        self.bob.phase = "LOCAL_QUANTUM_CORE_READY"
        self._event(ID_A, "LOCAL_QUANTUM_CORE_RESULT", result="ready", fields=sorted(self.alice.quantum_view))
        self._event(ID_B, "LOCAL_QUANTUM_CORE_RESULT", result="ready", fields=sorted(self.bob.quantum_view))

    def _ensure_own_ciphertext(self, state: PartyState) -> str:
        role = "A" if state.identity == ID_A else "B"
        if role in state.ciphertexts:
            return state.ciphertexts[role]
        if "tilde_own" not in state.quantum_view:
            raise ProtocolViolation("dialogue_requires_local_quantum_result")
        otp = _key_bits(state.keys.otp_for(state.identity), state.message_bits)
        values = [state.quantum_view["tilde_own"], otp]
        if state.mode == RELEASE:
            if "own_mask" not in state.private_local:
                state.private_local["own_mask"] = _key_bits(secrets.token_bytes(math.ceil(state.message_bits / 8)), state.message_bits)
            values.append(state.own_mask())
        ciphertext = xor_bits(*values)
        state.ciphertexts[role] = ciphertext
        return ciphertext

    def send_dialogue_b_qpass(self) -> bytes:
        state = self.bob
        self._require_active(state)
        if not state.detection_passed or state.next_send_seq != 1 or not state.decoy_accepted:
            raise ProtocolViolation("bob_dialogue_send_gate_closed")
        ciphertext = self._ensure_own_ciphertext(state)
        if state.ack_lengths is None:
            raise ProtocolViolation("bob_ack_context_unavailable")
        tq = compute_final_tq(
            sid=state.sid,
            params_hash=state.params_hash,
            length_s3=state.ack_lengths[0],
            length_s4=state.ack_lengths[1],
            h_ack=state.public_hashes["h_ack"],
            h_dist=state.public_hashes["h_dist"],
        )
        payload = {"decision": "QPASS", "ciphertext_bits": state.message_bits, "ciphertext": ciphertext}
        before = state.snapshot()
        raw = _build_record(state, "DIALOGUE_B_QPASS", payload, tq, 1)
        state.t_q = tq
        state.dialogue_b_sent = True
        state.phase = "BOB_DIALOGUE_SENT"
        self._record_send(state, raw, "DIALOGUE_B_QPASS", before)
        return raw

    def send_dialogue_a(self) -> bytes:
        state = self.alice
        self._require_active(state)
        if not state.qpass_authenticated or state.t_q is None or state.next_send_seq != 1:
            raise ProtocolViolation("alice_dialogue_requires_verified_bob_qpass")
        ciphertext = self._ensure_own_ciphertext(state)
        payload = {"ciphertext_bits": state.message_bits, "ciphertext": ciphertext}
        before = state.snapshot()
        raw = _build_record(state, "DIALOGUE_A", payload, state.t_q, 1)
        state.dialogue_a_sent = True
        state.phase = "ALICE_DIALOGUE_SENT"
        self._record_send(state, raw, "DIALOGUE_A", before)
        return raw

    def send_commit_set(self, identity: str) -> bytes:
        state = self._party(identity)
        self._require_active(state)
        if state.mode != RELEASE or state.next_send_seq != 2 or set(state.ciphertexts) != {"A", "B"} or state.t_q is None:
            raise ProtocolViolation("commit_set_send_gate_closed")
        self._ensure_own_ciphertext(state)
        lengths = list(state.params["release_lengths"])
        mask = state.own_mask()
        chunks: list[str] = []
        offset = 0
        for length in lengths:
            chunks.append(mask[offset : offset + length])
            offset += length
        role = "A" if identity == ID_A else "B"
        salts = [secrets.token_bytes(int(state.params["salt_bits"]) // 8) for _ in chunks]
        commitments = [
            {
                "j": j,
                "u_j": len(chunk),
                "d_j": compute_commitment(
                    state=state,
                    sender=identity,
                    j=j,
                    u_j=len(chunk),
                    ciphertext=state.ciphertexts[role],
                    chunk=chunk,
                    salt=salts[j - 1],
                ),
            }
            for j, chunk in enumerate(chunks, 1)
        ]
        state.private_local["own_openings"] = {
            j: {"j": j, "u_j": len(chunks[j - 1]), "chunk": chunks[j - 1], "salt": salts[j - 1].hex()}
            for j in range(1, len(chunks) + 1)
        }
        payload = {
            "message_bits": state.message_bits,
            "lambda": state.params["lambda"],
            "r": state.params["r"],
            "ciphertext_hash": hash_ce1(state.ciphertexts[role]),
            "commitments": commitments,
        }
        before = state.snapshot()
        raw = _build_record(state, "COMMIT_SET", payload, state.t_q, 2)
        state.commit_set_sent = True
        state.public_hashes["own_commit_set_hash"] = hashlib.sha256(raw).hexdigest()
        state.phase = "COMMIT_SET_SENT"
        self._record_send(state, raw, "COMMIT_SET", before)
        return raw

    def first_leader(self) -> str:
        if self.alice.t_q is None:
            raise ProtocolViolation("leader_requires_tq")
        digest = hash_ce1(["IAQD-FIRST", self.sid, self.params_hash, self.alice.t_q])
        return ID_A if int(digest[:2], 16) % 2 == 0 else ID_B

    def leader_for_block(self, j: int) -> str:
        first = self.first_leader()
        return first if j % 2 == 1 else peer_for(first)

    def _check_open_send_gate(self, state: PartyState, j: int) -> None:
        if state.mode != RELEASE or not state.peer_commit_set or not state.commit_set_sent:
            raise ProtocolViolation("opening_requires_both_commit_sets_locally")
        if j != len(state.own_open_sent) + 1 or state.next_send_seq != 2 + j:
            raise ProtocolViolation("opening_not_next_local_block")
        leader = self.leader_for_block(j)
        if state.identity == leader:
            if len(state.peer_open_accepted) != j - 1:
                raise ProtocolViolation("leader_waits_for_previous_peer_blocks")
        elif len(state.peer_open_accepted) != j:
            raise ProtocolViolation("follower_waits_for_current_leader_block")
        u_j = int(state.params["release_lengths"][j - 1])
        if state.S + u_j > state.V + int(state.params["lambda"]):
            raise ProtocolViolation("bounded_release_guard_closed")

    def send_open(self, identity: str, j: int) -> bytes:
        state = self._party(identity)
        self._require_active(state)
        self._check_open_send_gate(state, j)
        opening = copy.deepcopy(state.private_local["own_openings"][j])
        opening["commit_set_hash"] = state.public_hashes["own_commit_set_hash"]
        before = state.snapshot()
        raw = _build_record(state, "OPEN", opening, str(state.t_q), 2 + j)
        # S changes only after the actual wire send has succeeded.
        state.own_open_sent.add(j)
        state.S += int(opening["u_j"])
        state.max_gap = max(state.max_gap, state.S - state.V)
        state.phase = "OPEN_SENT"
        self._record_send(state, raw, "OPEN", before)
        return raw

    def _confirm_send_gate(self, state: PartyState) -> None:
        if state.recovered_peer_message is None or set(state.ciphertexts) != {"A", "B"} or state.t_q is None:
            raise ProtocolViolation("confirmation_requires_complete_local_recovery")
        if state.confirm_sent:
            raise ProtocolViolation("confirmation_already_sent")
        if state.mode == RELEASE and (
            state.S != state.message_bits
            or state.V != state.message_bits
            or len(state.own_open_sent) != state.params["r"]
            or len(state.peer_open_accepted) != state.params["r"]
        ):
            raise ProtocolViolation("release_confirmation_requires_all_local_open_events")

    def send_confirm(self, identity: str) -> bytes:
        state = self._party(identity)
        self._require_active(state)
        self._confirm_send_gate(state)
        expected_seq = 2 if state.mode == BASIC else int(state.params["r"]) + 3
        if state.next_send_seq != expected_seq:
            raise ProtocolViolation("confirmation_sequence_not_ready")
        payload = {"message_bits": state.message_bits, "confirm_label": "FULL-RECOVERY"}
        before = state.snapshot()
        raw = _build_record(state, "CONFIRM", payload, str(state.t_q), expected_seq)
        state.confirm_sent = True
        state.phase = "CONFIRM_SENT"
        self._record_send(state, raw, "CONFIRM", before)
        if state.peer_confirm_accepted:
            state.active = False
            state.terminated = True
            state.completed = True
            state.terminal_reason = "COMPLETED"
            state.phase = "COMPLETED"
        return raw

    def local_timeout(self, identity: str, waiting_for: str) -> None:
        state = self._party(identity)
        if state.terminated:
            return
        before = state.snapshot()
        state.active = False
        state.terminated = True
        state.terminal_reason = f"TIMEOUT_{waiting_for}"
        state.phase = "TERMINATED"
        self._event(
            identity,
            "LOCAL_TIMEOUT",
            result="terminated",
            waiting_for=waiting_for,
            state_before=before,
            state_after=state.snapshot(),
        )

    def local_abort(self, identity: str, reason: str = "PARTICIPANT_ABORT") -> None:
        state = self._party(identity)
        if state.terminated:
            return
        before = state.snapshot()
        state.active = False
        state.terminated = True
        state.terminal_reason = reason
        state.phase = "TERMINATED"
        self._event(identity, "LOCAL_ABORT", result="terminated", reason=reason, state_before=before, state_after=state.snapshot())

    def run_prelude(self, *, detection_errors: int = 0, detection_trials: int | None = None) -> bool:
        if not self.send_quantum():
            return False
        ack = self.send_qrecv_ack()
        if not self.deliver(ack, ID_A).accepted:
            return False
        decoy = self.send_decoy_info()
        if not self.deliver(decoy, ID_B).accepted:
            return False
        if not self.bob_detection(errors=detection_errors, trials=detection_trials):
            return False
        self.perform_quantum_core()
        dialogue_b = self.send_dialogue_b_qpass()
        if not self.deliver(dialogue_b, ID_A).accepted:
            return False
        dialogue_a = self.send_dialogue_a()
        if not self.deliver(dialogue_a, ID_B).accepted:
            return False
        return True

    def run_honest(self) -> ExecutionResult:
        if not self.run_prelude():
            return self.result()
        if self.params["mode"] == RELEASE:
            commit_a = self.send_commit_set(ID_A)
            commit_b = self.send_commit_set(ID_B)
            if not self.deliver(commit_a, ID_B).accepted or not self.deliver(commit_b, ID_A).accepted:
                return self.result()
            for j in range(1, int(self.params["r"]) + 1):
                leader = self.leader_for_block(j)
                follower = peer_for(leader)
                first = self.send_open(leader, j)
                if not self.deliver(first, follower).accepted:
                    return self.result()
                second = self.send_open(follower, j)
                if not self.deliver(second, leader).accepted:
                    return self.result()
        confirm_b = self.send_confirm(ID_B)
        confirm_a = self.send_confirm(ID_A)
        self.deliver(confirm_b, ID_A)
        self.deliver(confirm_a, ID_B)
        return self.result()

    def result(self) -> ExecutionResult:
        success = self.alice.completed and self.bob.completed
        if success:
            terminal = "COMPLETED"
        elif self.alice.terminal_reason or self.bob.terminal_reason:
            terminal = f"A:{self.alice.terminal_reason or 'ACTIVE'}|B:{self.bob.terminal_reason or 'ACTIVE'}"
        else:
            terminal = "IN_PROGRESS"
        records = [parse_record(raw) for raw in self.wire_records if _is_parseable_record(raw)]
        auth_kinds = {"MAC": 0, "PRF": 0}
        for record in records:
            auth_kinds[str(record["auth"]["kind"])] += 1
        mode = str(self.params["mode"])
        r = int(self.params["r"])
        return ExecutionResult(
            mode=mode,
            sid=self.sid,
            success=success,
            terminal_state=terminal,
            alice_state=self.alice.snapshot(),
            bob_state=self.bob.snapshot(),
            recovered_by_alice=self.alice.recovered_peer_message,
            recovered_by_bob=self.bob.recovered_peer_message,
            event_log=copy.deepcopy(self.event_log),
            wire_records=list(self.wire_records),
            metrics={
                "post_ke_record_count": len(records),
                "mac_record_count": auth_kinds["MAC"],
                "prf_confirm_count": auth_kinds["PRF"],
                "qtx_count": 1 if any(event["event"] == "QUANTUM_SEND" for event in self.event_log) else 0,
                "r": r,
                "causal_layers_core_spec": 5 if mode == BASIC else 6 + 2 * r,
                "records_core_spec": 6 if mode == BASIC else 8 + 2 * r,
            },
        )


def _is_parseable_record(raw: bytes) -> bool:
    try:
        parse_record(raw)
    except (TypeError, ValueError):
        return False
    return True


def state_json(state: PartyState) -> str:
    return json.dumps(state.snapshot(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

