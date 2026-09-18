from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import secrets
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence


PROTOCOL_VERSION = "IAQD-V6"
ID_A = "Alice"
ID_B = "Bob"
KE_MODE = "ideal_test_fixture"
KDF_INSTANTIATION = "benchmark_reference_only"
COMMITMENT_INSTANTIATION = "benchmark_reference_only"
GAMMA_FIELD_NAMES = ("sid", "role_label", "message", "ciphertext_a", "ciphertext_b", "T_Q")
OUTER_HEADER_FIELDS = (
    "ver",
    "sid",
    "sender",
    "receiver",
    "direction",
    "type",
    "seq",
    "params",
    "T_Q",
)
CONFIRMATION_PAYLOAD_FIELDS = ("confirmation",)
FORBIDDEN_CONFIRMATION_FIELDS = {
    "message",
    "message_hex",
    "message_base64",
    "plaintext",
    "recovered_message",
    "candidate",
    "secret",
}

FINAL_TQ_DOMAIN = "IAQD/final-quantum-transcript/V6"
PRE_TQ_DOMAIN = "IAQD/pre-quantum-context/V6"
DIST_DOMAIN = "IAQD/decoy-disclosure/V6"


def _u32(value: int) -> bytes:
    return int(value).to_bytes(4, "big", signed=False)


def canonical_encode(value: Any) -> bytes:
    """Typed, length-delimited deterministic encoding used by the reference model."""
    if value is None:
        return b"N" + _u32(0)
    if isinstance(value, bool):
        return b"T" + _u32(1) + (b"1" if value else b"0")
    if isinstance(value, int):
        payload = str(value).encode("ascii")
        return b"I" + _u32(len(payload)) + payload
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical floats must be finite")
        payload = format(value, ".17g").encode("ascii")
        return b"F" + _u32(len(payload)) + payload
    if isinstance(value, bytes):
        return b"B" + _u32(len(value)) + value
    if isinstance(value, str):
        payload = value.encode("utf-8")
        return b"S" + _u32(len(payload)) + payload
    if isinstance(value, (list, tuple)):
        payload = b"".join(canonical_encode(item) for item in value)
        return b"L" + _u32(len(value)) + _u32(len(payload)) + payload
    if isinstance(value, dict):
        pieces = []
        for key in sorted(value):
            if not isinstance(key, str):
                raise TypeError("canonical dictionary keys must be strings")
            pieces.append(canonical_encode(key))
            pieces.append(canonical_encode(value[key]))
        payload = b"".join(pieces)
        return b"D" + _u32(len(value)) + _u32(len(payload)) + payload
    raise TypeError(f"unsupported canonical type: {type(value)!r}")


def canonical_fields(fields: Sequence[tuple[str, Any]]) -> bytes:
    return canonical_encode([{"name": name, "value": value} for name, value in fields])


def xor_bytes(left: bytes, right: bytes) -> bytes:
    if len(left) != len(right):
        raise ValueError("xor operands must have equal length")
    return bytes(a ^ b for a, b in zip(left, right))


def pack_bits(bits: str) -> bytes:
    if any(bit not in "01" for bit in bits):
        raise ValueError("bit string contains a non-binary symbol")
    return len(bits).to_bytes(4, "big") + (int(bits or "0", 2).to_bytes((len(bits) + 7) // 8, "big"))


def unpack_bits(payload: bytes) -> str:
    length = int.from_bytes(payload[:4], "big")
    body = payload[4:]
    if length == 0:
        return ""
    return f"{int.from_bytes(body, 'big'):0{length}b}"[-length:]


def xor_bits(*values: str) -> str:
    if not values:
        return ""
    if len({len(value) for value in values}) != 1:
        raise ValueError("bit strings must have equal length")
    return "".join(str(sum(int(value[index]) for value in values) % 2) for index in range(len(values[0])))


def split_bits(bits: str, lambda_bits: int) -> list[str]:
    if lambda_bits <= 0:
        raise ValueError("lambda_bits must be positive")
    return [bits[offset : offset + lambda_bits] for offset in range(0, len(bits), lambda_bits)]


def reference_quantum_block(cluster_type: str, alice_message: str, bob_message: str) -> dict[str, Any]:
    """Deterministic non-E4 fixture following equations (6)-(7), never a joint 6n state."""
    if cluster_type not in {"phi1", "phi2"}:
        raise ValueError("unknown cluster type")
    label12 = "00" if cluster_type == "phi1" else "01"
    m_b = "00"
    m_a = "01"
    tilde_m_b = xor_bits(m_b, bob_message)
    tilde_m_a = xor_bits(m_a, alice_message)
    return {
        "cluster_type": cluster_type,
        "bell_labels": [label12, tilde_m_b, tilde_m_a],
        "decoded_intermediates": {
            "label12": label12,
            "M_B": m_b,
            "tilde_M_B": tilde_m_b,
            "M_A": m_a,
            "tilde_M_A": tilde_m_a,
            "alice_base56": m_a,
            "bob_base34": m_b,
        },
        "decoded_for_alice": bob_message,
        "decoded_for_bob": alice_message,
        "source": "reference_fixture_for_non_E4_workload",
    }


def assert_quantum_block_semantics(
    block: dict[str, Any],
    alice_message: str,
    bob_message: str,
) -> dict[str, str]:
    """Check the paper's M_A/M_B algebra independently from the executor outputs."""
    labels = [str(value) for value in block["bell_labels"]]
    if len(labels) != 3:
        raise AssertionError("a six-particle block must expose three Bell labels")
    label12, label34, label56 = labels
    intermediates = block.get("decoded_intermediates")
    if not isinstance(intermediates, dict):
        raise AssertionError("decoded_intermediates missing")
    expected = {
        "M_B": xor_bits(label34, bob_message),
        "tilde_M_B": label34,
        "M_A": xor_bits(label56, alice_message),
        "tilde_M_A": label56,
    }
    for name, value in expected.items():
        if str(intermediates.get(name)) != value:
            raise AssertionError(f"{name} disagrees with the independent Bell label algebra")
    if xor_bits(expected["M_B"], expected["tilde_M_B"]) != bob_message:
        raise AssertionError("Bob-direction message relation failed")
    if xor_bits(expected["M_A"], expected["tilde_M_A"]) != alice_message:
        raise AssertionError("Alice-direction message relation failed")
    return {"label12": label12, **expected}


def deterministic_bytes(seed: int, domain: str, length: int) -> bytes:
    output = bytearray()
    counter = 0
    key = hashlib.sha256(canonical_fields((("seed", seed), ("domain", domain)))).digest()
    while len(output) < length:
        output.extend(hmac.new(key, domain.encode("utf-8") + counter.to_bytes(4, "big"), hashlib.sha256).digest())
        counter += 1
    return bytes(output[:length])


def build_complete_decoy_disclosure(seed: int, n_blocks: int, ell: int) -> dict[str, Any]:
    """Materialize the complete two-sequence BB84 disclosure sent on the wire."""
    if n_blocks <= 0 or ell <= 0:
        raise ValueError("n_blocks and ell must be positive")
    population = n_blocks + ell

    def typed_list(element_type: str, values: list[Any]) -> dict[str, Any]:
        return {"length": len(values), "element_type": element_type, "values": values}

    disclosure: dict[str, Any] = {}
    for sequence in ("S3", "S4"):
        ranked = sorted(
            range(population),
            key=lambda index: hashlib.sha256(
                canonical_fields(
                    (("seed", seed), ("domain", "complete-decoy-disclosure"), ("sequence", sequence), ("index", index))
                )
            ).digest(),
        )
        positions = sorted(ranked[:ell])
        symbols = deterministic_bytes(seed, f"decoy-symbols:{sequence}:{n_blocks}:{ell}", ell)
        bases = ["Z" if value & 1 == 0 else "X" for value in symbols]
        states = [int((value >> 1) & 1) for value in symbols]
        disclosure[f"positions_{sequence}"] = typed_list("uint32", positions)
        disclosure[f"bases_{sequence}"] = typed_list("BB84_basis", bases)
        disclosure[f"states_{sequence}"] = typed_list("BB84_bit", states)
    return disclosure


def complete_decoy_disclosure_is_valid(disclosure: Any, n_blocks: int, ell: int) -> bool:
    expected = {
        "positions_S3": "uint32",
        "bases_S3": "BB84_basis",
        "states_S3": "BB84_bit",
        "positions_S4": "uint32",
        "bases_S4": "BB84_basis",
        "states_S4": "BB84_bit",
    }
    if not isinstance(disclosure, dict) or set(disclosure) != set(expected):
        return False
    population = n_blocks + ell
    for name, element_type in expected.items():
        item = disclosure.get(name)
        if not isinstance(item, dict) or set(item) != {"length", "element_type", "values"}:
            return False
        values = item.get("values")
        if item.get("element_type") != element_type or item.get("length") != ell or not isinstance(values, list) or len(values) != ell:
            return False
        if name.startswith("positions_"):
            if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 or value >= population for value in values):
                return False
            if len(set(values)) != ell:
                return False
        elif name.startswith("bases_") and any(value not in {"Z", "X"} for value in values):
            return False
        elif name.startswith("states_") and any(value not in {0, 1} for value in values):
            return False
    return True


@dataclass
class OperationCounters:
    """Runtime counters for executed cryptographic and protocol operations."""

    counts: Counter[str] = field(default_factory=Counter)
    kdf_labels: Counter[str] = field(default_factory=Counter)
    round_schedule: list[dict[str, Any]] = field(default_factory=list)

    def increment(self, name: str, amount: int = 1) -> None:
        self.counts[name] += int(amount)

    def record_kdf(self, label: str) -> None:
        self.increment("kdf_derivation_calls")
        self.kdf_labels[label] += 1

    def schedule_round(self, round_index: int, phase: str, send_events: Sequence[str]) -> None:
        if any(item["round_index"] == round_index for item in self.round_schedule):
            raise ValueError(f"upper-layer round {round_index} already scheduled")
        self.round_schedule.append(
            {
                "round_index": int(round_index),
                "phase": phase,
                "send_events": list(send_events),
            }
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "counts": dict(sorted(self.counts.items())),
            "kdf_labels": dict(sorted(self.kdf_labels.items())),
            "round_schedule": sorted(self.round_schedule, key=lambda item: item["round_index"]),
        }


def compute_sid(
    proto_ver: str,
    id_a: str,
    id_b: str,
    nonce_a: bytes,
    nonce_b: bytes,
    params: dict[str, Any],
    counters: OperationCounters | None = None,
) -> str:
    """Equation (20), encoded with explicit field names/types/lengths."""
    if counters is not None:
        counters.increment("session_identifier_hash_calls")
    return hashlib.sha256(
        canonical_fields(
            (
                ("proto_ver", proto_ver),
                ("ID_A", id_a),
                ("ID_B", id_b),
                ("N_A", nonce_a),
                ("N_B", nonce_b),
                ("params", params),
            )
        )
    ).hexdigest()


def derive_initial_leader(sid: str) -> str:
    return "Alice" if int(sid[:2], 16) % 2 == 0 else "Bob"


def leader_for_round(sid: str, round_index: int) -> str:
    initial = derive_initial_leader(sid)
    if round_index % 2 == 0:
        return initial
    return "Bob" if initial == "Alice" else "Alice"


def make_sid_for_leader(seed: int, requested: str, params: dict[str, Any]) -> tuple[str, bytes, bytes]:
    for attempt in range(4096):
        nonce_a = deterministic_bytes(seed + attempt, "nonce-a", 16)
        nonce_b = deterministic_bytes(seed + attempt, "nonce-b", 16)
        sid = compute_sid(PROTOCOL_VERSION, ID_A, ID_B, nonce_a, nonce_b, params)
        if derive_initial_leader(sid) == requested:
            return sid, nonce_a, nonce_b
    raise RuntimeError("could not construct requested sid leader")


@dataclass
class SessionKeys:
    mac_a: bytes
    mac_b: bytes
    conf_a: bytes
    conf_b: bytes
    otp_a: bytes
    otp_b: bytes
    ack: bytes
    dist: bytes
    counters: OperationCounters = field(default_factory=OperationCounters, repr=False)

    def mac_for_sender(self, sender: str) -> bytes:
        return self.mac_a if sender == "Alice" else self.mac_b

    def conf_for_role(self, role: str) -> bytes:
        return self.conf_a if role == "A" else self.conf_b


def derive_session_keys(
    shared_fixture: bytes,
    sid: str,
    message_bytes: int,
    counters: OperationCounters | None = None,
) -> SessionKeys:
    counters = counters or OperationCounters()

    def derive(label: str, length: int = 32) -> bytes:
        counters.record_kdf(label)
        return hmac.new(shared_fixture, canonical_fields((("domain", label), ("sid", sid))), hashlib.sha256).digest()[:length]

    return SessionKeys(
        mac_a=derive("IAQD/mac/A"),
        mac_b=derive("IAQD/mac/B"),
        conf_a=derive("IAQD/conf/A"),
        conf_b=derive("IAQD/conf/B"),
        otp_a=derive("IAQD/otp/A", message_bytes),
        otp_b=derive("IAQD/otp/B", message_bytes),
        ack=derive("IAQD/ack/B2A"),
        dist=derive("IAQD/dist/A2B"),
        counters=counters,
    )


def compute_gamma(
    role: str,
    key: bytes,
    sid: str,
    message: bytes,
    ciphertext_a: bytes,
    ciphertext_b: bytes,
    quantum_transcript_digest: str,
    counters: OperationCounters | None = None,
    operation: str = "generate",
) -> str:
    if role not in {"A", "B"}:
        raise ValueError("role must be A or B")
    material = canonical_fields(
        (
            ("sid", sid),
            ("role_label", f"{role}-confirm"),
            ("message", message),
            ("ciphertext_a", ciphertext_a),
            ("ciphertext_b", ciphertext_b),
            ("T_Q", quantum_transcript_digest),
        )
    )
    if operation not in {"generate", "verify"}:
        raise ValueError("gamma operation must be generate or verify")
    if counters is not None:
        counters.increment(f"message_confirmation_gamma_{operation}_calls")
    return hmac.new(key, material, hashlib.sha256).hexdigest()


def compute_public_message_hash(message: bytes) -> str:
    """Original-AQD public equality/dictionary oracle used only by its dedicated ablation."""
    return hashlib.sha256(canonical_fields((("message", message),))).hexdigest()


def compute_commitment(
    sid: str,
    sender: str,
    block_index: int,
    mask_chunk: str,
    salt: bytes,
    counters: OperationCounters | None = None,
    operation: str = "generate",
) -> str:
    if len(salt) < 16:
        raise ValueError("commitment salt must be at least 128 bits")
    material = canonical_fields(
        (
            ("domain", "IAQD/fair-mask-commitment/V6"),
            ("sid", sid),
            ("P", sender),
            ("block_index", block_index),
            ("R_P_j", mask_chunk),
            ("salt", salt),
        )
    )
    if operation not in {"generate", "verify"}:
        raise ValueError("commitment operation must be generate or verify")
    if counters is not None:
        counters.increment(f"commitment_{operation}_calls")
    return hashlib.sha3_256(material).hexdigest()


def compute_qrecv_ack(
    key_ack: bytes,
    sid: str,
    id_b: str,
    id_a: str,
    recv_marker: str,
    length_s3: int,
    length_s4: int,
    counters: OperationCounters | None = None,
    operation: str = "generate",
) -> str:
    """Equation (24): tau_ack = MAC_Kack(sid||ID_B||ID_A||QRECV||L3||L4)."""
    material = canonical_fields(
        (
            ("sid", sid),
            ("ID_B", id_b),
            ("ID_A", id_a),
            ("recv", recv_marker),
            ("L3", length_s3),
            ("L4", length_s4),
        )
    )
    if operation not in {"generate", "verify"}:
        raise ValueError("QRECV acknowledgement operation must be generate or verify")
    if counters is not None:
        counters.increment(f"qrecv_ack_tag_{operation}_calls")
    return hmac.new(key_ack, material, hashlib.sha256).hexdigest()


def decoy_auth_header(hdr: dict[str, Any], qrecv_record_id: str) -> dict[str, Any]:
    """Canonical hdr_dist for equation tau_dist = MAC_Kdist(hdr_dist || D_decoy)."""
    return {
        "ver": hdr["ver"],
        "sid": hdr["sid"],
        "ID_A": "Alice",
        "ID_B": "Bob",
        "direction": hdr["direction"],
        "type": hdr["type"],
        "seq": hdr["seq"],
        "params": hdr["params"],
        "pre_quantum_context_digest": hdr["T_Q"],
        "qrecv_record_id": qrecv_record_id,
    }


def compute_tau_dist(
    key_dist: bytes,
    hdr_dist: dict[str, Any],
    decoy_disclosure: dict[str, Any],
    counters: OperationCounters | None = None,
    operation: str = "generate",
) -> str:
    required_header = {
        "ver", "sid", "ID_A", "ID_B", "direction", "type", "seq", "params",
        "pre_quantum_context_digest", "qrecv_record_id",
    }
    if set(hdr_dist) != required_header:
        raise ValueError("hdr_dist schema mismatch")
    if operation not in {"generate", "verify"}:
        raise ValueError("tau_dist operation must be generate or verify")
    material = canonical_fields(
        (
            ("domain", DIST_DOMAIN),
            ("hdr_dist", hdr_dist),
            ("D_decoy", decoy_disclosure),
        )
    )
    if counters is not None:
        counters.increment(f"tau_dist_{operation}_calls")
    return hmac.new(key_dist, material, hashlib.sha256).hexdigest()


def compute_final_tq(fields: dict[str, Any], counters: OperationCounters | None = None) -> str:
    required = {
        "proto_ver", "sid", "ID_A", "ID_B", "L3", "L4", "qrecv_record_id",
        "qrecv_digest", "decoy_record_digest", "measured_qber", "qber_threshold",
        "decision", "params",
    }
    if set(fields) != required:
        raise ValueError(f"final T_Q fields differ: missing={required-set(fields)}, extra={set(fields)-required}")
    if counters is not None:
        counters.increment("final_quantum_transcript_hash_calls")
    return hashlib.sha256(
        canonical_fields(
            (
                ("domain", FINAL_TQ_DOMAIN),
                ("quantum_transcript", {name: fields[name] for name in sorted(fields)}),
            )
        )
    ).hexdigest()


@dataclass
class ValidationPolicy:
    bind_sid: bool = True
    bind_direction: bool = True
    enforce_sequence: bool = True
    require_receipt_ack: bool = True
    use_record_mac: bool = True
    use_keyed_confirmation: bool = True
    require_dist_auth: bool = True
    public_record_hash: bool = False


@dataclass
class ReceiverState:
    sid: str
    protocol_version: str
    local_identity: str
    peer_identity: str
    receive_direction: str
    params: dict[str, Any]
    pre_quantum_context_digest: str
    _final_quantum_transcript_digest: str | None = field(default=None, init=False, repr=False)
    phase: str = "INITIALIZED"
    expected_record_type: str | None = None
    expected_sequence: dict[str, int] = field(default_factory=lambda: {"A2B": 0, "B2A": 0})
    accepted_record_ids: set[str] = field(default_factory=set)
    verified_qrecv: bool = False
    verified_qrecv_record_id: str | None = None
    decoy_disclosure_allowed: bool = False
    processed_confirmation_records: set[str] = field(default_factory=set)
    terminated: bool = False
    terminal_reason: str | None = None
    locally_recovered_peer_message: bytes | None = None
    ciphertext_a: bytes | None = None
    ciphertext_b: bytes | None = None
    confirmation_accepted: bool = False
    commitments: dict[int, dict[str, Any]] = field(default_factory=dict)
    opened_blocks: set[int] = field(default_factory=set)
    next_open_block: int = 0
    verified_mask_bits: str = ""
    key_references: dict[str, bytes] = field(default_factory=dict, repr=False)
    logical_erasure: bool = False
    policy: ValidationPolicy = field(default_factory=ValidationPolicy)

    def snapshot(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("_final_quantum_transcript_digest", None)
        value["final_quantum_transcript_digest"] = self.final_quantum_transcript_digest
        value["accepted_record_ids"] = sorted(self.accepted_record_ids)
        value["processed_confirmation_records"] = sorted(self.processed_confirmation_records)
        value["opened_blocks"] = sorted(self.opened_blocks)
        value["key_references"] = sorted(self.key_references)
        value["locally_recovered_peer_message"] = (
            None if self.locally_recovered_peer_message is None else self.locally_recovered_peer_message.hex()
        )
        value["ciphertext_a"] = None if self.ciphertext_a is None else self.ciphertext_a.hex()
        value["ciphertext_b"] = None if self.ciphertext_b is None else self.ciphertext_b.hex()
        return value

    def erase_keys_logically(self) -> None:
        self.key_references.clear()
        self.logical_erasure = True

    def transcript_digest_for(self, record_type: str) -> str:
        if record_type in {"QRECV", "DECOY_DISCLOSURE"}:
            return self.pre_quantum_context_digest
        if self.final_quantum_transcript_digest is None:
            raise RuntimeError("final_quantum_transcript_digest_unavailable")
        return self.final_quantum_transcript_digest

    @property
    def final_quantum_transcript_digest(self) -> str | None:
        return self._final_quantum_transcript_digest

    def freeze_final_quantum_transcript(self, digest: str) -> None:
        if self._final_quantum_transcript_digest is not None:
            raise RuntimeError("final_quantum_transcript_digest_already_frozen")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("invalid final quantum transcript digest")
        self._final_quantum_transcript_digest = digest


@dataclass
class ValidationResult:
    accepted: bool
    reason: str
    record_id: str | None
    trace: list[str]


def _wire_encode(record: dict[str, Any]) -> bytes:
    return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def parse_record(raw_record_bytes: bytes) -> dict[str, Any]:
    decoded = json.loads(raw_record_bytes.decode("utf-8"))
    if not isinstance(decoded, dict) or set(decoded) != {"hdr", "payload", "record_mac"}:
        raise ValueError("invalid top-level record schema")
    if not isinstance(decoded["hdr"], dict) or tuple(decoded["hdr"].keys()) != tuple(sorted(OUTER_HEADER_FIELDS)):
        if set(decoded["hdr"]) != set(OUTER_HEADER_FIELDS):
            raise ValueError("invalid header schema")
    if not isinstance(decoded["payload"], dict) or not isinstance(decoded["record_mac"], str):
        raise ValueError("invalid payload or tag type")
    return decoded


def record_auth_material(hdr: dict[str, Any], payload: dict[str, Any]) -> bytes:
    return canonical_fields((("hdr", hdr), ("payload", payload)))


def create_record(
    *,
    sid: str,
    sender: str,
    receiver: str,
    direction: str,
    record_type: str,
    seq: int,
    params: dict[str, Any],
    quantum_transcript_digest: str,
    payload: dict[str, Any],
    session_keys: SessionKeys,
    policy: ValidationPolicy | None = None,
) -> bytes:
    policy = policy or ValidationPolicy()
    hdr = {
        "ver": PROTOCOL_VERSION,
        "sid": sid,
        "sender": sender,
        "receiver": receiver,
        "direction": direction,
        "type": record_type,
        "seq": seq,
        "params": params,
        "T_Q": quantum_transcript_digest,
    }
    material = record_auth_material(hdr, payload)
    if policy.public_record_hash:
        tag = hashlib.sha256(material).hexdigest()
        session_keys.counters.increment("public_record_hash_generation_calls")
    else:
        tag = hmac.new(session_keys.mac_for_sender(sender), material, hashlib.sha256).hexdigest()
        if record_type == "DIALOGUE_CIPHERTEXT":
            session_keys.counters.increment("paper_dialogue_record_mac_generation_calls")
        elif record_type == "COMMITMENT":
            session_keys.counters.increment("commitment_record_mac_generation_calls")
        elif record_type == "OPENING":
            session_keys.counters.increment("opening_record_mac_generation_calls")
        else:
            session_keys.counters.increment("engineering_outer_record_mac_generation_calls")
    return _wire_encode({"hdr": hdr, "payload": payload, "record_mac": tag})


def _reject(state: ReceiverState, reason: str, trace: list[str]) -> ValidationResult:
    terminal_by_reason = {
        "old_or_replayed_sequence": "REJECT_SEQUENCE",
        "future_sequence": "REJECT_SEQUENCE",
        "record_already_accepted": "REJECT_SEQUENCE",
        "confirmation_context_unavailable": "REJECT_CONTEXT",
        "confirmation_gamma_invalid": "REJECT_CONFIRMATION",
    }
    state.terminal_reason = terminal_by_reason.get(reason, "REJECT_AUTH")
    return ValidationResult(False, reason, None, trace + [f"reject:{reason}"])


def _contains_forbidden_key(value: Any) -> bool:
    if isinstance(value, dict):
        return any(key.lower() in FORBIDDEN_CONFIRMATION_FIELDS or _contains_forbidden_key(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_forbidden_key(item) for item in value)
    return False


def validate_record(
    raw_record_bytes: bytes,
    receiver_state: ReceiverState,
    session_keys: SessionKeys,
) -> ValidationResult:
    """Three-input stateful record validator."""
    trace = ["parse_raw_bytes"]
    try:
        record = parse_record(raw_record_bytes)
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError, TypeError) as error:
        return _reject(receiver_state, f"parse_error:{type(error).__name__}", trace)
    hdr = record["hdr"]
    payload = record["payload"]
    policy = receiver_state.policy

    trace.append("check_version")
    if hdr["ver"] != receiver_state.protocol_version:
        return _reject(receiver_state, "version_mismatch", trace)
    trace.append("check_sid")
    if policy.bind_sid and hdr["sid"] != receiver_state.sid:
        return _reject(receiver_state, "sid_mismatch", trace)
    trace.append("check_identity_and_direction")
    if hdr["sender"] != receiver_state.peer_identity or hdr["receiver"] != receiver_state.local_identity:
        return _reject(receiver_state, "identity_mismatch", trace)
    if policy.bind_direction and hdr["direction"] != receiver_state.receive_direction:
        return _reject(receiver_state, "direction_mismatch", trace)
    trace.append("check_type_sequence_and_phase")
    if receiver_state.terminated:
        return _reject(receiver_state, "session_already_terminated", trace)
    if receiver_state.expected_record_type is not None and hdr["type"] != receiver_state.expected_record_type:
        return _reject(receiver_state, "unexpected_type_or_phase", trace)
    expected_seq = receiver_state.expected_sequence.get(hdr["direction"], 0)
    if policy.enforce_sequence and hdr["seq"] < expected_seq:
        return _reject(receiver_state, "old_or_replayed_sequence", trace)
    if policy.enforce_sequence and hdr["seq"] > expected_seq:
        return _reject(receiver_state, "future_sequence", trace)
    trace.append("check_params_and_T_Q")
    if hdr["params"] != receiver_state.params:
        return _reject(receiver_state, "params_mismatch", trace)
    try:
        expected_tq = receiver_state.transcript_digest_for(str(hdr["type"]))
    except RuntimeError:
        return _reject(receiver_state, "final_T_Q_unavailable", trace)
    if hdr["T_Q"] != expected_tq:
        return _reject(receiver_state, "T_Q_mismatch", trace)

    trace.append("verify_outer_record_authentication")
    material = record_auth_material(hdr, payload)
    expected_mac = (
        hashlib.sha256(material).hexdigest()
        if policy.public_record_hash
        else hmac.new(session_keys.mac_for_sender(hdr["sender"]), material, hashlib.sha256).hexdigest()
    )
    if policy.public_record_hash:
        session_keys.counters.increment("public_record_hash_verification_calls")
    elif hdr["type"] == "DIALOGUE_CIPHERTEXT":
        session_keys.counters.increment("paper_dialogue_record_mac_verification_calls")
    elif hdr["type"] == "COMMITMENT":
        session_keys.counters.increment("commitment_record_mac_verification_calls")
    elif hdr["type"] == "OPENING":
        session_keys.counters.increment("opening_record_mac_verification_calls")
    else:
        session_keys.counters.increment("engineering_outer_record_mac_verification_calls")
    if policy.use_record_mac and not hmac.compare_digest(record["record_mac"], expected_mac):
        return _reject(receiver_state, "record_mac_invalid", trace)
    record_id = hashlib.sha256(raw_record_bytes).hexdigest()
    if record_id in receiver_state.accepted_record_ids:
        return _reject(receiver_state, "record_already_accepted", trace)

    record_type = hdr["type"]
    if record_type == "QRECV":
        required_qrecv = {"recv", "L3", "L4", "tau_ack"}
        if set(payload) != required_qrecv:
            return _reject(receiver_state, "qrecv_schema_invalid", trace)
        expected_ack = compute_qrecv_ack(
            session_keys.ack,
            receiver_state.sid,
            receiver_state.peer_identity,
            receiver_state.local_identity,
            str(payload["recv"]),
            int(payload["L3"]),
            int(payload["L4"]),
            session_keys.counters,
            "verify",
        )
        if not hmac.compare_digest(str(payload["tau_ack"]), expected_ack):
            return _reject(receiver_state, "qrecv_ack_invalid", trace)
        receiver_state.verified_qrecv = True
        receiver_state.verified_qrecv_record_id = record_id
        receiver_state.decoy_disclosure_allowed = True
    elif record_type == "DECOY_DISCLOSURE":
        if not receiver_state.verified_qrecv or not receiver_state.decoy_disclosure_allowed:
            return _reject(receiver_state, "qrecv_not_verified", trace)
        if policy.require_receipt_ack and payload.get("qrecv_record_id") != receiver_state.verified_qrecv_record_id:
            return _reject(receiver_state, "qrecv_receipt_ack_invalid", trace)
        required_decoy = {"qrecv_record_id", "D_decoy", "tau_dist"}
        if set(payload) != required_decoy or not isinstance(payload.get("D_decoy"), dict):
            return _reject(receiver_state, "decoy_disclosure_schema_invalid", trace)
        ell = int(receiver_state.params.get("decoys_per_sequence", 0))
        n_blocks = int(receiver_state.params.get("n_blocks", 0))
        if not complete_decoy_disclosure_is_valid(payload["D_decoy"], n_blocks, ell):
            return _reject(receiver_state, "decoy_disclosure_schema_invalid", trace)
        if policy.require_dist_auth:
            trace.append("verify_tau_dist")
            hdr_dist = decoy_auth_header(hdr, str(payload["qrecv_record_id"]))
            expected_dist = compute_tau_dist(
                session_keys.dist,
                hdr_dist,
                payload["D_decoy"],
                session_keys.counters,
                "verify",
            )
            if not hmac.compare_digest(str(payload["tau_dist"]), expected_dist):
                return _reject(receiver_state, "tau_dist_invalid", trace)
    elif record_type == "COMMITMENT":
        try:
            block_index = int(payload["block_index"])
            commitment = str(payload["commitment"])
        except (KeyError, TypeError, ValueError):
            return _reject(receiver_state, "commitment_schema_invalid", trace)
        receiver_state.commitments[block_index] = {
            "commitment": commitment,
            "sender": hdr["sender"],
            "direction": hdr["direction"],
        }
    elif record_type == "OPENING":
        try:
            block_index = int(payload["block_index"])
            mask_chunk = str(payload["mask_chunk"])
            salt = bytes.fromhex(str(payload["salt"]))
        except (KeyError, TypeError, ValueError):
            return _reject(receiver_state, "opening_schema_invalid", trace)
        if block_index in receiver_state.opened_blocks:
            return _reject(receiver_state, "duplicate_opening", trace)
        if block_index != receiver_state.next_open_block:
            return _reject(receiver_state, "out_of_order_opening", trace)
        commitment_entry = receiver_state.commitments.get(block_index)
        if commitment_entry is None:
            return _reject(receiver_state, "opening_without_commitment", trace)
        if commitment_entry["sender"] != hdr["sender"] or commitment_entry["direction"] != hdr["direction"]:
            return _reject(receiver_state, "opening_context_mismatch", trace)
        actual = compute_commitment(
            receiver_state.sid,
            hdr["sender"],
            block_index,
            mask_chunk,
            salt,
            session_keys.counters,
            "verify",
        )
        if not hmac.compare_digest(actual, commitment_entry["commitment"]):
            return _reject(receiver_state, "commitment_opening_mismatch", trace)
        receiver_state.opened_blocks.add(block_index)
        receiver_state.next_open_block += 1
        receiver_state.verified_mask_bits += mask_chunk
    elif record_type == "MESSAGE_CONFIRMATION":
        trace.append("check_confirmation_payload_schema")
        if set(payload) != set(CONFIRMATION_PAYLOAD_FIELDS) or _contains_forbidden_key(payload):
            return _reject(receiver_state, "confirmation_payload_schema_invalid", trace)
        trace.append("check_local_recovery_context")
        if (
            receiver_state.locally_recovered_peer_message is None
            or receiver_state.ciphertext_a is None
            or receiver_state.ciphertext_b is None
            or receiver_state.final_quantum_transcript_digest is None
        ):
            return _reject(receiver_state, "confirmation_context_unavailable", trace)
        trace.append("recompute_inner_gamma_from_local_message")
        peer_role = "A" if hdr["sender"] == "Alice" else "B"
        expected_gamma = compute_gamma(
            peer_role,
            session_keys.conf_for_role(peer_role),
            receiver_state.sid,
            receiver_state.locally_recovered_peer_message,
            receiver_state.ciphertext_a,
            receiver_state.ciphertext_b,
            receiver_state.final_quantum_transcript_digest,
            session_keys.counters,
            "verify",
        )
        if policy.use_keyed_confirmation and not hmac.compare_digest(str(payload["confirmation"]), expected_gamma):
            return _reject(receiver_state, "confirmation_gamma_invalid", trace)
        receiver_state.confirmation_accepted = True
        receiver_state.processed_confirmation_records.add(record_id)

    receiver_state.accepted_record_ids.add(record_id)
    receiver_state.expected_sequence[hdr["direction"]] = expected_seq + 1
    trace.append("accept_and_update_state")
    return ValidationResult(True, "accepted", record_id, trace)


def mutate_record(raw: bytes, path: Sequence[str], value: Any, session_keys: SessionKeys | None = None) -> bytes:
    record = parse_record(raw)
    target: Any = record
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    if session_keys is not None:
        hdr, payload = record["hdr"], record["payload"]
        record["record_mac"] = hmac.new(
            session_keys.mac_for_sender(hdr["sender"]), record_auth_material(hdr, payload), hashlib.sha256
        ).hexdigest()
    return _wire_encode(record)


@dataclass
class ExecutionResult:
    mode: str
    sid: str
    terminal_state: str
    success: bool
    logical_erasure: bool
    recovered_by_alice: bytes | None
    recovered_by_bob: bytes | None
    gamma_a_valid: bool
    gamma_b_valid: bool
    trace: list[dict[str, Any]]
    wire_records: list[bytes]
    metrics: dict[str, Any]
    fairness_prefixes: list[dict[str, Any]] = field(default_factory=list)
    masks: dict[str, str] = field(default_factory=dict)
    commitments: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    quantum_inputs: list[dict[str, Any]] = field(default_factory=list)


class IAQDReferenceExecutor:
    """Executable upper-layer IAQD reference model; KE and KDF are explicit test fixtures."""

    def __init__(
        self,
        seed: int,
        *,
        measured_qber: float = 0.0,
        qber_threshold: float = 0.11,
        decoys_per_sequence: int = 4,
    ):
        self.seed = int(seed)
        self.measured_qber = float(measured_qber)
        self.qber_threshold = float(qber_threshold)
        if int(decoys_per_sequence) <= 0:
            raise ValueError("decoys_per_sequence must be positive")
        self.decoys_per_sequence = int(decoys_per_sequence)
        self.counters = OperationCounters()

    @staticmethod
    def _append(trace: list[dict[str, Any]], event: str, **details: Any) -> None:
        trace.append({"index": len(trace), "event": event, **details})

    def _setup(
        self,
        mode: str,
        n_blocks: int,
        lambda_bits: int | None,
        requested_leader: str | None,
        extra_params: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any], SessionKeys, ReceiverState, ReceiverState, list[dict[str, Any]]]:
        self.counters = OperationCounters()
        params = {
            "mode": mode,
            "n_blocks": n_blocks,
            "message_length_bits": 2 * n_blocks,
            "lambda_bits": lambda_bits,
            "KE_mode": KE_MODE,
            "kdf_instantiation": KDF_INSTANTIATION,
            "decoys_per_sequence": self.decoys_per_sequence,
        }
        params.update(extra_params or {})
        if requested_leader is None:
            nonce_a = deterministic_bytes(self.seed, "nonce-a", 16)
            nonce_b = deterministic_bytes(self.seed, "nonce-b", 16)
            sid = compute_sid(PROTOCOL_VERSION, ID_A, ID_B, nonce_a, nonce_b, params, self.counters)
        else:
            sid, nonce_a, nonce_b = make_sid_for_leader(self.seed, requested_leader, params)
            self.counters.increment("session_identifier_hash_calls")
        trace: list[dict[str, Any]] = []
        self._append(trace, "session_initialization", nonce_a=nonce_a.hex(), nonce_b=nonce_b.hex())
        self._append(trace, "parameter_negotiation", params=params)
        self.counters.schedule_round(1, "session_and_parameter_negotiation", ["N_A_and_params", "N_B_and_params"])
        self._append(trace, "sid_construction", sid=sid)
        shared_fixture = deterministic_bytes(self.seed, f"ideal-ke:{sid}", 32)
        self._append(trace, "ideal_reference_ke", KE_mode=KE_MODE)
        message_bytes = max(1, (2 * n_blocks + 7) // 8)
        keys = derive_session_keys(shared_fixture, sid, message_bytes, self.counters)
        self._append(trace, "domain_separated_key_derivation", instantiation=KDF_INSTANTIATION)
        self.counters.increment("pre_quantum_context_hash_calls")
        pre_tq = hashlib.sha256(
            canonical_fields((("domain", PRE_TQ_DOMAIN), ("sid", sid), ("phase", "pre-detection")))
        ).hexdigest()
        alice = ReceiverState(
            sid=sid,
            protocol_version=PROTOCOL_VERSION,
            local_identity="Alice",
            peer_identity="Bob",
            receive_direction="B2A",
            params=params,
            pre_quantum_context_digest=pre_tq,
            phase="AWAIT_QRECV",
            expected_record_type="QRECV",
            key_references={"mac": keys.mac_a, "conf": keys.conf_a, "ack": keys.ack, "dist": keys.dist},
        )
        bob = ReceiverState(
            sid=sid,
            protocol_version=PROTOCOL_VERSION,
            local_identity="Bob",
            peer_identity="Alice",
            receive_direction="A2B",
            params=params,
            pre_quantum_context_digest=pre_tq,
            phase="QRECV_SENT",
            key_references={"mac": keys.mac_b, "conf": keys.conf_b, "ack": keys.ack, "dist": keys.dist},
        )
        return sid, params, keys, alice, bob, trace

    def _deliver(
        self,
        trace: list[dict[str, Any]],
        records: list[bytes],
        receiver: ReceiverState,
        raw: bytes,
        keys: SessionKeys,
        event: str,
    ) -> ValidationResult:
        before = receiver.snapshot()
        result = validate_record(raw, receiver, keys)
        records.append(raw)
        self._append(
            trace,
            event,
            accepted=result.accepted,
            reason=result.reason,
            raw_record_sha256=hashlib.sha256(raw).hexdigest(),
            receiver_before=before,
            validation_trace=result.trace,
            receiver_after=receiver.snapshot(),
        )
        return result

    def _quantum_and_detection(
        self,
        sid: str,
        params: dict[str, Any],
        keys: SessionKeys,
        alice: ReceiverState,
        bob: ReceiverState,
        trace: list[dict[str, Any]],
        records: list[bytes],
        quantum_blocks: Sequence[dict[str, Any]] | None = None,
    ) -> str:
        quantum_blocks = list(quantum_blocks or [])
        cluster_sequence = [str(block.get("cluster_type")) for block in quantum_blocks]
        self._append(trace, "cluster_preparation", n_blocks=params["n_blocks"], cluster_sequence=cluster_sequence)
        decoys_per_sequence = int(params.get("decoys_per_sequence", 4))
        length_s3 = int(params["n_blocks"]) + decoys_per_sequence
        length_s4 = int(params["n_blocks"]) + decoys_per_sequence
        self._append(trace, "quantum_delivery", transmitted_cluster_qubits=2 * params["n_blocks"], L3=length_s3, L4=length_s4)
        tau_ack = compute_qrecv_ack(
            keys.ack, sid, ID_B, ID_A, "QRECV", length_s3, length_s4,
            keys.counters, "generate",
        )
        self.counters.schedule_round(2, "authenticated_quantum_receipt", ["Bob_to_Alice_QRECV"])
        qrecv = create_record(
            sid=sid,
            sender="Bob",
            receiver="Alice",
            direction="B2A",
            record_type="QRECV",
            seq=0,
            params=params,
            quantum_transcript_digest=alice.pre_quantum_context_digest,
            payload={"recv": "QRECV", "L3": length_s3, "L4": length_s4, "tau_ack": tau_ack},
            session_keys=keys,
        )
        qrecv_result = self._deliver(trace, records, alice, qrecv, keys, "authenticated_qrecv")
        if not qrecv_result.accepted:
            raise AssertionError(qrecv_result.reason)
        bob.verified_qrecv = True
        bob.verified_qrecv_record_id = qrecv_result.record_id
        bob.decoy_disclosure_allowed = True
        bob.expected_record_type = "DECOY_DISCLOSURE"
        decoy_disclosure = build_complete_decoy_disclosure(
            self.seed,
            int(params["n_blocks"]),
            decoys_per_sequence,
        )
        decoy_header = {
            "ver": PROTOCOL_VERSION,
            "sid": sid,
            "sender": "Alice",
            "receiver": "Bob",
            "direction": "A2B",
            "type": "DECOY_DISCLOSURE",
            "seq": 0,
            "params": params,
            "T_Q": bob.pre_quantum_context_digest,
        }
        tau_dist = compute_tau_dist(
            keys.dist,
            decoy_auth_header(decoy_header, str(qrecv_result.record_id)),
            decoy_disclosure,
            keys.counters,
            "generate",
        )
        self.counters.schedule_round(3, "authenticated_decoy_disclosure", ["Alice_to_Bob_DECOY_DISCLOSURE"])
        decoy = create_record(
            sid=sid,
            sender="Alice",
            receiver="Bob",
            direction="A2B",
            record_type="DECOY_DISCLOSURE",
            seq=0,
            params=params,
            quantum_transcript_digest=bob.pre_quantum_context_digest,
            payload={
                "qrecv_record_id": qrecv_result.record_id,
                "D_decoy": decoy_disclosure,
                "tau_dist": tau_dist,
            },
            session_keys=keys,
        )
        decoy_result = self._deliver(trace, records, bob, decoy, keys, "authenticated_decoy_disclosure")
        if not decoy_result.accepted:
            raise AssertionError(decoy_result.reason)
        decision = "accept" if self.measured_qber <= self.qber_threshold else "abort"
        self._append(trace, "qber_decision", qber=self.measured_qber, qber_threshold=self.qber_threshold, decision=decision)
        qrecv_digest = hashlib.sha256(qrecv).hexdigest()
        decoy_digest = hashlib.sha256(decoy).hexdigest()
        tq_fields = {
            "proto_ver": PROTOCOL_VERSION,
            "sid": sid,
            "ID_A": ID_A,
            "ID_B": ID_B,
            "L3": length_s3,
            "L4": length_s4,
            "qrecv_record_id": qrecv_result.record_id,
            "qrecv_digest": qrecv_digest,
            "decoy_record_digest": decoy_digest,
            "measured_qber": self.measured_qber,
            "qber_threshold": self.qber_threshold,
            "decision": decision,
            "params": params,
        }
        tq = compute_final_tq(tq_fields, self.counters)
        alice.freeze_final_quantum_transcript(tq)
        bob.freeze_final_quantum_transcript(tq)
        self._append(trace, "T_Q_construction", T_Q=tq, bound_fields=tq_fields, frozen=True)
        if decision != "accept":
            alice.terminated = bob.terminated = True
            alice.terminal_reason = bob.terminal_reason = "ABORT_QBER"
        return tq

    def run_basic(
        self,
        message_a_bits: str,
        message_b_bits: str,
        *,
        quantum_blocks: Sequence[dict[str, Any]] | None = None,
    ) -> ExecutionResult:
        if len(message_a_bits) != len(message_b_bits) or len(message_a_bits) % 2:
            raise ValueError("basic messages must have equal even bit length")
        n_blocks = len(message_a_bits) // 2
        supplied_blocks = [dict(block) for block in (quantum_blocks or [])]
        if supplied_blocks and len(supplied_blocks) != n_blocks:
            raise ValueError("one externally decoded quantum block is required per message block")
        if not supplied_blocks:
            supplied_blocks = [
                reference_quantum_block(
                    "phi1",
                    message_a_bits[index * 2 : index * 2 + 2],
                    message_b_bits[index * 2 : index * 2 + 2],
                )
                for index in range(n_blocks)
            ]
        cluster_sequence = [str(block["cluster_type"]) for block in supplied_blocks]
        sid, params, keys, alice, bob, trace = self._setup(
            "basic", n_blocks, None, None, {"cluster_sequence": cluster_sequence}
        )
        records: list[bytes] = []
        tq = self._quantum_and_detection(sid, params, keys, alice, bob, trace, records, supplied_blocks)
        if alice.terminal_reason == "ABORT_QBER":
            alice.erase_keys_logically()
            bob.erase_keys_logically()
            self._append(trace, "terminal", state="ABORT_QBER", logical_erasure=True)
            return ExecutionResult(
                mode="basic", sid=sid, terminal_state="ABORT_QBER", success=False, logical_erasure=True,
                recovered_by_alice=None, recovered_by_bob=None, gamma_a_valid=False, gamma_b_valid=False,
                trace=trace, wire_records=records, metrics=self._metrics(records, trace, n_blocks, fair=False),
                quantum_inputs=supplied_blocks,
            )
        width = max(1, (len(message_a_bits) + 7) // 8)
        message_a = int(message_a_bits, 2).to_bytes(width, "big")
        message_b = int(message_b_bits, 2).to_bytes(width, "big")
        algebra_blocks = [
            assert_quantum_block_semantics(
                block,
                message_a_bits[index * 2 : index * 2 + 2],
                message_b_bits[index * 2 : index * 2 + 2],
            )
            for index, block in enumerate(supplied_blocks)
        ]
        m_a_bits = "".join(block["M_A"] for block in algebra_blocks)
        m_b_bits = "".join(block["M_B"] for block in algebra_blocks)
        tilde_m_a_bits = "".join(block["tilde_M_A"] for block in algebra_blocks)
        tilde_m_b_bits = "".join(block["tilde_M_B"] for block in algebra_blocks)
        m_a = int(m_a_bits, 2).to_bytes(width, "big")
        m_b = int(m_b_bits, 2).to_bytes(width, "big")
        tilde_m_a = int(tilde_m_a_bits, 2).to_bytes(width, "big")
        tilde_m_b = int(tilde_m_b_bits, 2).to_bytes(width, "big")
        ciphertext_a = xor_bytes(tilde_m_a, keys.otp_a)
        ciphertext_b = xor_bytes(tilde_m_b, keys.otp_b)
        self.counters.increment("bell_measurement_calls", 3 * n_blocks)
        self.counters.increment("pauli_operation_calls", 2 * n_blocks)
        self._append(
            trace,
            "consume_external_bell_decode",
            blocks=n_blocks,
            quantum_blocks=supplied_blocks,
            algebra_blocks=algebra_blocks,
            M_A=m_a_bits,
            tilde_M_A=tilde_m_a_bits,
            M_B=m_b_bits,
            tilde_M_B=tilde_m_b_bits,
            K_A_otp=keys.otp_a.hex(),
            K_B_otp=keys.otp_b.hex(),
            C_A=ciphertext_a.hex(),
            C_B=ciphertext_b.hex(),
        )

        self.counters.schedule_round(4, "protected_dialogue_ciphertexts", ["Alice_to_Bob_C_A", "Bob_to_Alice_C_B"])
        bob.phase = "DIALOGUE"
        bob.expected_record_type = "DIALOGUE_CIPHERTEXT"
        raw_a = create_record(
            sid=sid, sender="Alice", receiver="Bob", direction="A2B", record_type="DIALOGUE_CIPHERTEXT",
            seq=1, params=params, quantum_transcript_digest=tq, payload={"ciphertext": ciphertext_a.hex()}, session_keys=keys,
        )
        if not self._deliver(trace, records, bob, raw_a, keys, "protected_dialogue_record_A").accepted:
            raise AssertionError("Alice dialogue record rejected")
        alice.phase = "DIALOGUE"
        alice.expected_record_type = "DIALOGUE_CIPHERTEXT"
        raw_b = create_record(
            sid=sid, sender="Bob", receiver="Alice", direction="B2A", record_type="DIALOGUE_CIPHERTEXT",
            seq=1, params=params, quantum_transcript_digest=tq, payload={"ciphertext": ciphertext_b.hex()}, session_keys=keys,
        )
        if not self._deliver(trace, records, alice, raw_b, keys, "protected_dialogue_record_B").accepted:
            raise AssertionError("Bob dialogue record rejected")
        decrypted_tilde_m_a = xor_bytes(ciphertext_a, keys.otp_a)
        decrypted_tilde_m_b = xor_bytes(ciphertext_b, keys.otp_b)
        recovered_by_bob = xor_bytes(m_a, decrypted_tilde_m_a)
        recovered_by_alice = xor_bytes(m_b, decrypted_tilde_m_b)
        alice.locally_recovered_peer_message = recovered_by_alice
        bob.locally_recovered_peer_message = recovered_by_bob
        alice.ciphertext_a = bob.ciphertext_a = ciphertext_a
        alice.ciphertext_b = bob.ciphertext_b = ciphertext_b
        self._append(
            trace,
            "message_recovery",
            Alice_uses="M_B xor tilde_M_B",
            Bob_uses="M_A xor tilde_M_A",
            recovered_by_alice=recovered_by_alice.hex(),
            recovered_by_bob=recovered_by_bob.hex(),
        )

        gamma_a = compute_gamma(
            "A", keys.conf_a, sid, message_a, ciphertext_a, ciphertext_b, tq,
            keys.counters, "generate",
        )
        self.counters.schedule_round(5, "keyed_message_confirmation", ["Alice_to_Bob_gamma_A", "Bob_to_Alice_gamma_B"])
        bob.phase = "CONFIRMATION"
        bob.expected_record_type = "MESSAGE_CONFIRMATION"
        confirm_a = create_record(
            sid=sid, sender="Alice", receiver="Bob", direction="A2B", record_type="MESSAGE_CONFIRMATION",
            seq=2, params=params, quantum_transcript_digest=tq, payload={"confirmation": gamma_a}, session_keys=keys,
        )
        result_a = self._deliver(trace, records, bob, confirm_a, keys, "keyed_confirmation_A")
        gamma_b = compute_gamma(
            "B", keys.conf_b, sid, message_b, ciphertext_a, ciphertext_b, tq,
            keys.counters, "generate",
        )
        alice.phase = "CONFIRMATION"
        alice.expected_record_type = "MESSAGE_CONFIRMATION"
        confirm_b = create_record(
            sid=sid, sender="Bob", receiver="Alice", direction="B2A", record_type="MESSAGE_CONFIRMATION",
            seq=2, params=params, quantum_transcript_digest=tq, payload={"confirmation": gamma_b}, session_keys=keys,
        )
        result_b = self._deliver(trace, records, alice, confirm_b, keys, "keyed_confirmation_B")
        decoded_a_bits = "".join(str(block["decoded_for_alice"]) for block in supplied_blocks)
        decoded_b_bits = "".join(str(block["decoded_for_bob"]) for block in supplied_blocks)
        quantum_decode_consistent = decoded_a_bits == message_b_bits and decoded_b_bits == message_a_bits
        success = (
            result_a.accepted and result_b.accepted and recovered_by_alice == message_b
            and recovered_by_bob == message_a and quantum_decode_consistent
        )
        terminal = "COMPLETED" if success else "REJECT_CONFIRMATION"
        alice.terminated = bob.terminated = True
        alice.terminal_reason = bob.terminal_reason = terminal
        alice.erase_keys_logically()
        bob.erase_keys_logically()
        self._append(trace, "terminal", state=terminal, logical_erasure=True)
        return ExecutionResult(
            mode="basic", sid=sid, terminal_state=terminal, success=success,
            logical_erasure=alice.logical_erasure and bob.logical_erasure,
            recovered_by_alice=recovered_by_alice, recovered_by_bob=recovered_by_bob,
            gamma_a_valid=result_a.accepted, gamma_b_valid=result_b.accepted,
            trace=trace, wire_records=records,
            metrics=self._metrics(records, trace, n_blocks, fair=False),
            quantum_inputs=supplied_blocks,
        )

    def run_fair(
        self,
        message_a_bits: str,
        message_b_bits: str,
        lambda_bits: int,
        requested_leader: str | None = None,
        abort_after_openings: int | None = None,
        cluster_sequence: Sequence[str] | None = None,
        quantum_blocks: Sequence[dict[str, Any]] | None = None,
    ) -> ExecutionResult:
        if len(message_a_bits) != len(message_b_bits) or len(message_a_bits) % 2:
            raise ValueError("fair messages must have equal even bit length")
        n_blocks = len(message_a_bits) // 2
        if lambda_bits > len(message_a_bits):
            raise ValueError("lambda_bits exceeds message length")
        sequence = list(cluster_sequence or [str(block["cluster_type"]) for block in (quantum_blocks or [])] or ["phi1"] * n_blocks)
        if len(sequence) != n_blocks or any(item not in {"phi1", "phi2"} for item in sequence):
            raise ValueError("cluster_sequence must contain one phi1/phi2 entry per block")
        supplied_blocks = [dict(block) for block in (quantum_blocks or [])]
        if supplied_blocks and len(supplied_blocks) != n_blocks:
            raise ValueError("one quantum block per fair message block is required")
        if not supplied_blocks:
            supplied_blocks = [
                reference_quantum_block(
                    sequence[index],
                    message_a_bits[index * 2 : index * 2 + 2],
                    message_b_bits[index * 2 : index * 2 + 2],
                )
                for index in range(n_blocks)
            ]
        if [str(block["cluster_type"]) for block in supplied_blocks] != sequence:
            raise ValueError("cluster_sequence disagrees with supplied quantum blocks")
        sid, params, keys, alice, bob, trace = self._setup(
            "fair",
            n_blocks,
            lambda_bits,
            requested_leader,
            {"cluster_sequence": sequence, "commitment_instantiation": COMMITMENT_INSTANTIATION},
        )
        records: list[bytes] = []
        tq = self._quantum_and_detection(sid, params, keys, alice, bob, trace, records, supplied_blocks)
        if alice.terminal_reason == "ABORT_QBER":
            alice.erase_keys_logically()
            bob.erase_keys_logically()
            self._append(trace, "terminal", state="ABORT_QBER", logical_erasure=True)
            return ExecutionResult(
                mode="fair", sid=sid, terminal_state="ABORT_QBER", success=False, logical_erasure=True,
                recovered_by_alice=None, recovered_by_bob=None, gamma_a_valid=False, gamma_b_valid=False,
                trace=trace, wire_records=records, metrics=self._metrics(records, trace, n_blocks, fair=True),
                quantum_inputs=supplied_blocks,
            )
        bit_length = len(message_a_bits)
        mask_a = "".join(f"{byte:08b}" for byte in deterministic_bytes(self.seed, f"mask-a:{sid}", (bit_length + 7) // 8))[-bit_length:]
        mask_b = "".join(f"{byte:08b}" for byte in deterministic_bytes(self.seed, f"mask-b:{sid}", (bit_length + 7) // 8))[-bit_length:]
        otp_a_bits = "".join(f"{byte:08b}" for byte in keys.otp_a)[-bit_length:]
        otp_b_bits = "".join(f"{byte:08b}" for byte in keys.otp_b)[-bit_length:]
        algebra_blocks = [
            assert_quantum_block_semantics(
                block,
                message_a_bits[index * 2 : index * 2 + 2],
                message_b_bits[index * 2 : index * 2 + 2],
            )
            for index, block in enumerate(supplied_blocks)
        ]
        m_a_bits = "".join(block["M_A"] for block in algebra_blocks)
        m_b_bits = "".join(block["M_B"] for block in algebra_blocks)
        tilde_m_a_bits = "".join(block["tilde_M_A"] for block in algebra_blocks)
        tilde_m_b_bits = "".join(block["tilde_M_B"] for block in algebra_blocks)
        ciphertext_a_bits = xor_bits(tilde_m_a_bits, otp_a_bits, mask_a)
        ciphertext_b_bits = xor_bits(tilde_m_b_bits, otp_b_bits, mask_b)
        ciphertext_a = pack_bits(ciphertext_a_bits)
        ciphertext_b = pack_bits(ciphertext_b_bits)
        self.counters.increment("bell_measurement_calls", 3 * n_blocks)
        self.counters.increment("pauli_operation_calls", 2 * n_blocks)
        self._append(
            trace,
            "fair_ciphertexts",
            algebra_blocks=algebra_blocks,
            M_A=m_a_bits,
            tilde_M_A=tilde_m_a_bits,
            M_B=m_b_bits,
            tilde_M_B=tilde_m_b_bits,
            K_A_otp_bits=otp_a_bits,
            K_B_otp_bits=otp_b_bits,
            R_A=mask_a,
            R_B=mask_b,
            C_A=ciphertext_a.hex(),
            C_B=ciphertext_b.hex(),
            masks_fresh=True,
        )

        self.counters.schedule_round(4, "fair_ciphertexts", ["Alice_to_Bob_C_A_fair", "Bob_to_Alice_C_B_fair"])
        bob.phase = "FAIR_CIPHERTEXT"
        bob.expected_record_type = "DIALOGUE_CIPHERTEXT"
        raw_a = create_record(
            sid=sid, sender="Alice", receiver="Bob", direction="A2B", record_type="DIALOGUE_CIPHERTEXT",
            seq=1, params=params, quantum_transcript_digest=tq, payload={"ciphertext": ciphertext_a.hex()}, session_keys=keys,
        )
        self._deliver(trace, records, bob, raw_a, keys, "fair_ciphertext_record_A")
        alice.phase = "FAIR_CIPHERTEXT"
        alice.expected_record_type = "DIALOGUE_CIPHERTEXT"
        raw_b = create_record(
            sid=sid, sender="Bob", receiver="Alice", direction="B2A", record_type="DIALOGUE_CIPHERTEXT",
            seq=1, params=params, quantum_transcript_digest=tq, payload={"ciphertext": ciphertext_b.hex()}, session_keys=keys,
        )
        self._deliver(trace, records, alice, raw_b, keys, "fair_ciphertext_record_B")

        chunks_a = split_bits(mask_a, lambda_bits)
        chunks_b = split_bits(mask_b, lambda_bits)
        salts_a = [deterministic_bytes(self.seed + index, f"salt-a:{sid}", 16) for index in range(len(chunks_a))]
        salts_b = [deterministic_bytes(self.seed + index, f"salt-b:{sid}", 16) for index in range(len(chunks_b))]
        commitments_a = [
            compute_commitment(sid, "Alice", index, chunk, salts_a[index], keys.counters, "generate")
            for index, chunk in enumerate(chunks_a)
        ]
        commitments_b = [
            compute_commitment(sid, "Bob", index, chunk, salts_b[index], keys.counters, "generate")
            for index, chunk in enumerate(chunks_b)
        ]
        self.counters.schedule_round(5, "authenticated_commitment_exchange", ["Alice_commitments", "Bob_commitments"])
        seq_a = seq_b = 2
        for index, commitment in enumerate(commitments_a):
            bob.phase = "COMMITMENTS"
            bob.expected_record_type = "COMMITMENT"
            raw = create_record(
                sid=sid, sender="Alice", receiver="Bob", direction="A2B", record_type="COMMITMENT", seq=seq_a,
                params=params, quantum_transcript_digest=tq, payload={"block_index": index, "commitment": commitment}, session_keys=keys,
            )
            self._deliver(trace, records, bob, raw, keys, "authenticated_commitment_A")
            seq_a += 1
        for index, commitment in enumerate(commitments_b):
            alice.phase = "COMMITMENTS"
            alice.expected_record_type = "COMMITMENT"
            raw = create_record(
                sid=sid, sender="Bob", receiver="Alice", direction="B2A", record_type="COMMITMENT", seq=seq_b,
                params=params, quantum_transcript_digest=tq, payload={"block_index": index, "commitment": commitment}, session_keys=keys,
            )
            self._deliver(trace, records, alice, raw, keys, "authenticated_commitment_B")
            seq_b += 1
        self._append(trace, "all_commitments_authenticated_before_opening", count=len(commitments_a) + len(commitments_b))

        fairness_prefixes: list[dict[str, Any]] = []
        la = lb = 0
        opening_count = 0
        aborted = False
        for round_index in range(len(chunks_a)):
            order = [leader_for_round(sid, round_index)]
            order.append("Bob" if order[0] == "Alice" else "Alice")
            self.counters.schedule_round(
                6 + round_index,
                f"fair_opening_block_{round_index}",
                [f"{order[0]}_opening_{round_index}", f"{order[1]}_opening_{round_index}"],
            )
            for sender in order:
                if abort_after_openings is not None and opening_count == abort_after_openings:
                    aborted = True
                    self._append(trace, "participant_abort", opening_prefix=opening_count, L_A=la, L_B=lb)
                    break
                if sender == "Alice":
                    receiver, direction, chunk, salt, seq = bob, "A2B", chunks_a[round_index], salts_a[round_index], seq_a
                    seq_a += 1
                else:
                    receiver, direction, chunk, salt, seq = alice, "B2A", chunks_b[round_index], salts_b[round_index], seq_b
                    seq_b += 1
                receiver.phase = "FAIR_OPEN"
                receiver.expected_record_type = "OPENING"
                raw = create_record(
                    sid=sid, sender=sender, receiver=receiver.local_identity, direction=direction, record_type="OPENING", seq=seq,
                    params=params, quantum_transcript_digest=tq,
                    payload={"block_index": round_index, "mask_chunk": chunk, "salt": salt.hex()}, session_keys=keys,
                )
                result = self._deliver(trace, records, receiver, raw, keys, f"verified_opening_{sender}")
                if not result.accepted:
                    raise AssertionError(result.reason)
                if sender == "Alice":
                    lb += len(chunk)
                else:
                    la += len(chunk)
                opening_count += 1
                prefix = {
                    "opening_prefix": opening_count,
                    "round_index": round_index,
                    "sender": sender,
                    "opened_bits": len(chunk),
                    "L_A": la,
                    "L_B": lb,
                    "information_lead": abs(la - lb),
                    "lambda_bits": lambda_bits,
                    "within_bound": abs(la - lb) <= lambda_bits,
                }
                fairness_prefixes.append(prefix)
                self._append(trace, "fairness_prefix", **prefix)
            if aborted:
                break

        if aborted:
            terminal = "ABORT_PARTICIPANT"
            alice.terminated = bob.terminated = True
            alice.terminal_reason = bob.terminal_reason = terminal
            alice.erase_keys_logically()
            bob.erase_keys_logically()
            self._append(trace, "terminal", state=terminal, logical_erasure=True)
            return ExecutionResult(
                mode="fair", sid=sid, terminal_state=terminal, success=False, logical_erasure=True,
                recovered_by_alice=None, recovered_by_bob=None, gamma_a_valid=False, gamma_b_valid=False,
                trace=trace, wire_records=records, metrics=self._metrics(records, trace, n_blocks, fair=True),
                fairness_prefixes=fairness_prefixes, masks={"R_A": mask_a, "R_B": mask_b},
                commitments={
                    "A": [{"block_index": i, "chunk": c, "salt": salts_a[i].hex(), "commitment": commitments_a[i]} for i, c in enumerate(chunks_a)],
                    "B": [{"block_index": i, "chunk": c, "salt": salts_b[i].hex(), "commitment": commitments_b[i]} for i, c in enumerate(chunks_b)],
                },
                quantum_inputs=supplied_blocks,
            )

        decrypted_tilde_m_b = xor_bits(unpack_bits(ciphertext_b), otp_b_bits, alice.verified_mask_bits)
        decrypted_tilde_m_a = xor_bits(unpack_bits(ciphertext_a), otp_a_bits, bob.verified_mask_bits)
        recovered_a_bits = xor_bits(m_b_bits, decrypted_tilde_m_b)
        recovered_b_bits = xor_bits(m_a_bits, decrypted_tilde_m_a)
        width = max(1, (bit_length + 7) // 8)
        recovered_by_alice = int(recovered_a_bits, 2).to_bytes(width, "big")
        recovered_by_bob = int(recovered_b_bits, 2).to_bytes(width, "big")
        message_a = int(message_a_bits, 2).to_bytes(width, "big")
        message_b = int(message_b_bits, 2).to_bytes(width, "big")
        alice.locally_recovered_peer_message = recovered_by_alice
        bob.locally_recovered_peer_message = recovered_by_bob
        alice.ciphertext_a = bob.ciphertext_a = ciphertext_a
        alice.ciphertext_b = bob.ciphertext_b = ciphertext_b
        self._append(trace, "fair_message_recovery_complete", L_A=la, L_B=lb)

        confirmation_round = 6 + len(chunks_a)
        self.counters.schedule_round(
            confirmation_round,
            "keyed_message_confirmation",
            ["Alice_to_Bob_gamma_A", "Bob_to_Alice_gamma_B"],
        )
        gamma_a = compute_gamma(
            "A", keys.conf_a, sid, message_a, ciphertext_a, ciphertext_b, tq,
            keys.counters, "generate",
        )
        bob.phase = "CONFIRMATION"
        bob.expected_record_type = "MESSAGE_CONFIRMATION"
        confirm_a = create_record(
            sid=sid, sender="Alice", receiver="Bob", direction="A2B", record_type="MESSAGE_CONFIRMATION", seq=seq_a,
            params=params, quantum_transcript_digest=tq, payload={"confirmation": gamma_a}, session_keys=keys,
        )
        result_a = self._deliver(trace, records, bob, confirm_a, keys, "fair_keyed_confirmation_A")
        gamma_b = compute_gamma(
            "B", keys.conf_b, sid, message_b, ciphertext_a, ciphertext_b, tq,
            keys.counters, "generate",
        )
        alice.phase = "CONFIRMATION"
        alice.expected_record_type = "MESSAGE_CONFIRMATION"
        confirm_b = create_record(
            sid=sid, sender="Bob", receiver="Alice", direction="B2A", record_type="MESSAGE_CONFIRMATION", seq=seq_b,
            params=params, quantum_transcript_digest=tq, payload={"confirmation": gamma_b}, session_keys=keys,
        )
        result_b = self._deliver(trace, records, alice, confirm_b, keys, "fair_keyed_confirmation_B")
        decoded_a_bits = "".join(str(block["decoded_for_alice"]) for block in supplied_blocks)
        decoded_b_bits = "".join(str(block["decoded_for_bob"]) for block in supplied_blocks)
        success = (
            recovered_by_alice == message_b
            and recovered_by_bob == message_a
            and decoded_a_bits == message_b_bits
            and decoded_b_bits == message_a_bits
            and result_a.accepted
            and result_b.accepted
            and all(prefix["within_bound"] for prefix in fairness_prefixes)
        )
        terminal = "COMPLETED" if success else "REJECT_CONFIRMATION"
        alice.terminated = bob.terminated = True
        alice.terminal_reason = bob.terminal_reason = terminal
        alice.erase_keys_logically()
        bob.erase_keys_logically()
        self._append(trace, "terminal", state=terminal, logical_erasure=True)
        return ExecutionResult(
            mode="fair", sid=sid, terminal_state=terminal, success=success, logical_erasure=True,
            recovered_by_alice=recovered_by_alice, recovered_by_bob=recovered_by_bob,
            gamma_a_valid=result_a.accepted, gamma_b_valid=result_b.accepted,
            trace=trace, wire_records=records, metrics=self._metrics(records, trace, n_blocks, fair=True),
            fairness_prefixes=fairness_prefixes, masks={"R_A": mask_a, "R_B": mask_b},
            commitments={
                "A": [{"block_index": i, "chunk": c, "salt": salts_a[i].hex(), "commitment": commitments_a[i]} for i, c in enumerate(chunks_a)],
                "B": [{"block_index": i, "chunk": c, "salt": salts_b[i].hex(), "commitment": commitments_b[i]} for i, c in enumerate(chunks_b)],
            },
            quantum_inputs=supplied_blocks,
        )

    def _metrics(self, records: Sequence[bytes], trace: Sequence[dict[str, Any]], n_blocks: int, fair: bool) -> dict[str, Any]:
        parsed = [parse_record(raw) for raw in records]
        payload_bytes = sum(len(json.dumps(record["payload"], sort_keys=True, separators=(",", ":")).encode()) for record in parsed)
        header_bytes = sum(len(json.dumps(record["hdr"], sort_keys=True, separators=(",", ":")).encode()) for record in parsed)
        tag_bytes = sum(len(bytes.fromhex(record["record_mac"])) for record in parsed)
        snapshot = self.counters.snapshot()
        counts = snapshot["counts"]
        ell = self.decoys_per_sequence
        decoy_indexes = [index for index, record in enumerate(parsed) if record["hdr"]["type"] == "DECOY_DISCLOSURE"]
        if len(decoy_indexes) != 1:
            raise AssertionError("an honest execution must contain exactly one decoy disclosure record")
        decoy_index = decoy_indexes[0]
        decoy_payload = parsed[decoy_index]["payload"]["D_decoy"]
        actual_decoy_payload_bytes = len(
            json.dumps(decoy_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        )
        actual_decoy_record_bytes = len(records[decoy_index])

        def count(name: str) -> int:
            return int(counts.get(name, 0))

        paper_logical_generate = (
            count("qrecv_ack_tag_generate_calls")
            + count("tau_dist_generate_calls")
            + count("paper_dialogue_record_mac_generation_calls")
            + count("message_confirmation_gamma_generate_calls")
        )
        paper_logical_verify = (
            count("qrecv_ack_tag_verify_calls")
            + count("tau_dist_verify_calls")
            + count("paper_dialogue_record_mac_verification_calls")
            + count("message_confirmation_gamma_verify_calls")
        )
        record_mac_generation = (
            count("paper_dialogue_record_mac_generation_calls")
            + count("commitment_record_mac_generation_calls")
            + count("opening_record_mac_generation_calls")
            + count("engineering_outer_record_mac_generation_calls")
        )
        record_mac_verification = (
            count("paper_dialogue_record_mac_verification_calls")
            + count("commitment_record_mac_verification_calls")
            + count("opening_record_mac_verification_calls")
            + count("engineering_outer_record_mac_verification_calls")
        )
        actual_auth_generation = (
            count("qrecv_ack_tag_generate_calls")
            + count("tau_dist_generate_calls")
            + count("message_confirmation_gamma_generate_calls")
            + record_mac_generation
        )
        actual_auth_verification = (
            count("qrecv_ack_tag_verify_calls")
            + count("tau_dist_verify_calls")
            + count("message_confirmation_gamma_verify_calls")
            + record_mac_verification
        )
        return {
            "ell_per_sequence": ell,
            "prepared_qubits": 6 * n_blocks + 2 * ell,
            "transmitted_qubits": 2 * n_blocks + 2 * ell,
            "local_storage_qubits": 4 * n_blocks,
            "serialized_bytes": sum(len(raw) for raw in records),
            "actual_total_wire_bytes": sum(len(raw) for raw in records),
            "actual_decoy_payload_bytes": actual_decoy_payload_bytes,
            "actual_decoy_record_bytes": actual_decoy_record_bytes,
            "application_payload_bytes": payload_bytes,
            "header_bytes": header_bytes,
            "tag_bytes": tag_bytes,
            "paper_logical_auth_generation_count": paper_logical_generate,
            "paper_logical_auth_verification_count": paper_logical_verify,
            "record_mac_generation_count": record_mac_generation,
            "record_mac_verification_count": record_mac_verification,
            "engineering_outer_record_mac_generation_count": count("engineering_outer_record_mac_generation_calls"),
            "engineering_outer_record_mac_verification_count": count("engineering_outer_record_mac_verification_calls"),
            "commitment_record_mac_generation_count": count("commitment_record_mac_generation_calls"),
            "commitment_record_mac_verification_count": count("commitment_record_mac_verification_calls"),
            "opening_record_mac_generation_count": count("opening_record_mac_generation_calls"),
            "opening_record_mac_verification_count": count("opening_record_mac_verification_calls"),
            "paper_dialogue_record_mac_generation_count": count("paper_dialogue_record_mac_generation_calls"),
            "paper_dialogue_record_mac_verification_count": count("paper_dialogue_record_mac_verification_calls"),
            "QRECV_ack_MAC_generation_count": count("qrecv_ack_tag_generate_calls"),
            "QRECV_ack_MAC_verification_count": count("qrecv_ack_tag_verify_calls"),
            "tau_dist_generation_count": count("tau_dist_generate_calls"),
            "tau_dist_verification_count": count("tau_dist_verify_calls"),
            "gamma_generation_count": count("message_confirmation_gamma_generate_calls"),
            "gamma_verification_count": count("message_confirmation_gamma_verify_calls"),
            "commitment_generation_count": count("commitment_generate_calls"),
            "commitment_verification_count": count("commitment_verify_calls"),
            "opening_count": sum(record["hdr"]["type"] == "OPENING" for record in parsed),
            "KDF_derivation_count": count("kdf_derivation_calls"),
            "KDF_purpose_labels": snapshot["kdf_labels"],
            "session_identifier_hash_count": count("session_identifier_hash_calls"),
            "pre_quantum_context_hash_count": count("pre_quantum_context_hash_calls"),
            "final_quantum_transcript_hash_count": count("final_quantum_transcript_hash_calls"),
            "commitment_hash_count": count("commitment_generate_calls") + count("commitment_verify_calls"),
            "bell_measurement_count": count("bell_measurement_calls"),
            "data_qubit_measurement_count": 6 * n_blocks,
            "decoy_measurement_count": 2 * ell,
            "protocol_total_measurement_count": 6 * n_blocks + 2 * ell,
            "pauli_operation_count": count("pauli_operation_calls"),
            "reference_actual_auth_generation_count": actual_auth_generation,
            "reference_actual_auth_verification_count": actual_auth_verification,
            "reference_actual_auth_operation_count": actual_auth_generation + actual_auth_verification,
            "one_way_authenticated_record_count": len(records),
            "upper_layer_round_count": len(snapshot["round_schedule"]),
            "upper_layer_round_schedule": snapshot["round_schedule"],
            "KE_internal_round_count": "not_measured",
            "KE_internal_round_scope": "ideal_test_fixture_external_not_measured",
            "trace_events": len(trace),
            "fair_mode": fair,
        }


def timed_execution(callable_object: Any, repetitions: int) -> tuple[list[float], list[ExecutionResult]]:
    durations: list[float] = []
    results: list[ExecutionResult] = []
    for _ in range(repetitions):
        started = time.perf_counter()
        result = callable_object()
        durations.append(time.perf_counter() - started)
        results.append(result)
    return durations, results

