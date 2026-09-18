from __future__ import annotations

import hashlib
import math
import time
from collections import deque
from dataclasses import asdict, dataclass, replace

from scipy.optimize import brentq
from scipy.special import zeta

from common_v6 import (
    COUNTEREXAMPLES,
    LOGS,
    PRISM_DIR,
    PROCESSED,
    RAW,
    run_prism,
    runtime_config,
    sha256_file,
    summary_row,
    write_csv,
    write_json,
)


@dataclass(frozen=True)
class AbortState:
    phase: int = 3
    Pass_decoy: bool = True
    AliceKnowsB: bool = False
    BobKnowsA: bool = False
    aborted_by_alice: bool = False
    terminal: bool = False
    OutA_has_B: bool = False
    OutB_abort: bool = False
    BobWaitingForCA: bool = False
    timed_out: bool = False


def successors(state: AbortState, *, timeout_enabled: bool) -> list[tuple[str, AbortState]]:
    if state.terminal:
        return []
    result: list[tuple[str, AbortState]] = []
    if state.phase == 3 and state.Pass_decoy:
        result.append(("send_cb", replace(state, phase=5)))
    if state.phase == 5:
        result.append(("alice_recovers_b", replace(state, phase=6, AliceKnowsB=True, OutA_has_B=True)))
    if state.phase == 6 and state.AliceKnowsB:
        result.append(("continue", replace(state, phase=7)))
        if not state.BobKnowsA:
            result.append(("alice_abort", replace(state, phase=7, aborted_by_alice=True)))
    if state.phase == 7 and state.aborted_by_alice:
        result.append(("bob_waiting_for_ca", replace(state, phase=8, BobWaitingForCA=True)))
    elif state.phase == 7:
        result.append(("send_ca", replace(state, phase=8, BobKnowsA=True)))
    if state.phase == 8 and state.BobWaitingForCA:
        if timeout_enabled:
            result.append(("timeout", replace(state, phase=9, timed_out=True)))
        else:
            result.append(("wait_forever", state))
    elif state.phase == 8:
        result.append(("finish", replace(state, phase=10, terminal=True)))
    if state.phase == 9 and state.timed_out:
        result.append(("bob_abort", replace(state, phase=10, terminal=True, OutB_abort=True)))
    if timeout_enabled and state.phase == 8 and state.BobWaitingForCA:
        result.append(("delay", state))
    return result


def terminal_unfair(state: AbortState) -> bool:
    return state.terminal and state.timed_out and state.OutB_abort and state.AliceKnowsB and not state.BobKnowsA


def nonterminal_liveness_failure(state: AbortState) -> bool:
    return (
        not state.terminal and state.BobWaitingForCA and state.AliceKnowsB
        and not state.BobKnowsA and not state.OutB_abort
    )


def shortest_abort_trace(*, timeout_enabled: bool) -> list[dict[str, object]]:
    initial = AbortState()
    queue = deque([(initial, [{"state": asdict(initial)}])])
    visited = {initial}
    while queue:
        state, trace = queue.popleft()
        if (terminal_unfair(state) if timeout_enabled else nonterminal_liveness_failure(state)):
            return trace
        for action, target in successors(state, timeout_enabled=timeout_enabled):
            if target not in visited:
                visited.add(target)
                queue.append((target, trace + [{"action": action, "state": asdict(target)}]))
    raise AssertionError("requested abort target is unreachable")


def shifted_zipf_offset(min_entropy_bits: int, exponent: float) -> float:
    target = 2.0 ** (-min_entropy_bits)

    def objective(offset: float) -> float:
        return (offset + 1.0) ** (-exponent) / float(zeta(exponent, offset + 1.0)) - target

    upper = 1.0
    while objective(upper) > 0:
        upper *= 2
    return float(brentq(objective, 0.0, upper, xtol=1e-10, rtol=1e-12))


def shifted_zipf_recovery(q: int, offset: float, exponent: float) -> float:
    normalizer = float(zeta(exponent, offset + 1.0))
    tail = float(zeta(exponent, offset + q + 1.0))
    return max(0.0, min(1.0, (normalizer - tail) / normalizer))


def dictionary_timing(q: int) -> tuple[float, float]:
    started = time.perf_counter()
    table = {
        hashlib.sha256(f"synthetic-message-{rank:08d}".encode("ascii")).digest(): rank
        for rank in range(q)
    }
    preprocessing = time.perf_counter() - started
    target = hashlib.sha256(f"synthetic-message-{q - 1:08d}".encode("ascii")).digest()
    started = time.perf_counter()
    recovered = table.get(target)
    query = time.perf_counter() - started
    if recovered != q - 1:
        raise AssertionError("synthetic dictionary lookup failed")
    return preprocessing, query


def run() -> list[dict[str, object]]:
    config = runtime_config()["e1"]
    exponent = float(config["zipf_exponent"])
    raw_rows: list[dict[str, object]] = []
    messages = [b"AQD-controlled-message-A", b"AQD-controlled-message-B"]
    digests = [hashlib.sha256(message).hexdigest() for message in messages]
    if len(set(digests)) != 2:
        raise AssertionError("chosen-message pair collided")
    for chosen_bit, digest in enumerate(digests):
        guessed_bit = digests.index(digest)
        raw_rows.append(
            {
                "component": "chosen_message",
                "distribution": "two_equal_length_messages",
                "min_entropy_bits": 1,
                "q": 2,
                "recovery_rate": int(guessed_bit == chosen_bit),
                "chosen_bit": chosen_bit,
                "guessed_bit": guessed_bit,
                "privacy_advantage": 0.5,
                "session": "challenge",
                "digest": digest,
                "notes": "Deterministic analytic attack witness; no Monte Carlo.",
            }
        )
    repeated = b"AQD-repeated-cross-session-message"
    different = b"AQD-different-cross-session-message-"
    session_rows = [
        ("sid-A", repeated),
        ("sid-B", repeated),
        ("sid-C", different),
    ]
    equality = []
    for session, message in session_rows:
        digest = hashlib.sha256(message).hexdigest()
        equality.append(digest)
        raw_rows.append(
            {
                "component": "cross_session_equality",
                "distribution": "deterministic",
                "session": session,
                "digest": digest,
                "recovery_rate": "",
                "privacy_advantage": "",
                "notes": "Public unkeyed digest is session-independent.",
            }
        )
    if not (equality[0] == equality[1] and equality[0] != equality[2]):
        raise AssertionError("cross-session equality witness failed")

    for entropy in (4, 8, 12, 16, 20):
        support = 2**entropy
        q_values = sorted(
            {
                1,
                support,
                *[
                    max(1, min(support, int(math.ceil(support * float(fraction)))))
                    for fraction in config["dictionary_q_fractions"]
                ],
            }
        )
        timing_q = min(support, 65536)
        preprocessing, query = dictionary_timing(timing_q)
        for distribution in ("uniform", "zipf"):
            offset = shifted_zipf_offset(entropy, exponent) if distribution == "zipf" else math.nan
            for q in q_values:
                recovery = q / support if distribution == "uniform" else shifted_zipf_recovery(q, offset, exponent)
                raw_rows.append(
                    {
                        "component": "dictionary",
                        "distribution": distribution,
                        "min_entropy_bits": entropy,
                        "q": q,
                        "recovery_rate": recovery,
                        "zipf_exponent": exponent if distribution == "zipf" else "",
                        "zipf_offset": offset if distribution == "zipf" else "",
                        "preprocessing_seconds": preprocessing if q == timing_q else "",
                        "online_query_seconds": query if q == timing_q else "",
                        "notes": "Synthetic controlled distribution only; no real-business-distribution claim.",
                    }
                )

    hash_path = RAW / "e1_hash_dictionary_cross_session_v6.csv"
    write_csv(hash_path, raw_rows)
    timeout_trace = shortest_abort_trace(timeout_enabled=True)
    timeout_actions = [str(step["action"]) for step in timeout_trace if "action" in step]
    required_timeout_actions = [
        "send_cb", "alice_recovers_b", "alice_abort", "bob_waiting_for_ca", "timeout", "bob_abort"
    ]
    if timeout_actions != required_timeout_actions:
        raise AssertionError(f"unexpected timeout trace: {timeout_actions}")
    if not terminal_unfair(AbortState(**timeout_trace[-1]["state"])):
        raise AssertionError("timeout trace does not end in terminal unfair abort")
    no_timeout_trace = shortest_abort_trace(timeout_enabled=False)
    no_timeout_actions = [str(step["action"]) for step in no_timeout_trace if "action" in step]
    if no_timeout_actions != ["send_cb", "alice_recovers_b", "alice_abort", "bob_waiting_for_ca"]:
        raise AssertionError(f"unexpected no-timeout trace: {no_timeout_actions}")
    if not nonterminal_liveness_failure(AbortState(**no_timeout_trace[-1]["state"])):
        raise AssertionError("no-timeout trace is not a nonterminal liveness failure")

    states_path = COUNTEREXAMPLES / "e1_abort_terminal_all_states_v6.txt"
    transitions_path = COUNTEREXAMPLES / "e1_abort_terminal_all_transitions_v6.txt"
    strategy_path = COUNTEREXAMPLES / "e1_abort_terminal_optimal_strategy_v6.txt"
    prism_main = run_prism(
        PRISM_DIR / "original_aqd_abort_terminal_v6.nm",
        PRISM_DIR / "original_aqd_abort_terminal_v6.props",
        prop=1,
        log_path=LOGS / "e1_abort_terminal_prism_v6.log",
        strategy_path=strategy_path,
        states_path=states_path,
        transitions_path=transitions_path,
    )
    prism_output = run_prism(
        PRISM_DIR / "original_aqd_abort_terminal_v6.nm",
        PRISM_DIR / "original_aqd_abort_terminal_v6.props",
        prop=2,
        log_path=LOGS / "e1_abort_terminal_output_prism_v6.log",
    )
    if float(prism_main["result"]) != 1.0 or float(prism_output["result"]) != 1.0:
        raise AssertionError("timeout terminal abort property did not evaluate to one")
    no_timeout_main = run_prism(
        PRISM_DIR / "original_aqd_abort_no_timeout_v6.nm",
        PRISM_DIR / "original_aqd_abort_no_timeout_v6.props",
        prop=1,
        log_path=LOGS / "e1_abort_no_timeout_liveness_prism_v6.log",
        states_path=COUNTEREXAMPLES / "e1_abort_no_timeout_all_states_v6.txt",
        transitions_path=COUNTEREXAMPLES / "e1_abort_no_timeout_all_transitions_v6.txt",
    )
    no_timeout_terminal = run_prism(
        PRISM_DIR / "original_aqd_abort_no_timeout_v6.nm",
        PRISM_DIR / "original_aqd_abort_no_timeout_v6.props",
        prop=2,
        log_path=LOGS / "e1_abort_no_timeout_no_bob_abort_prism_v6.log",
    )
    if float(no_timeout_main["result"]) != 1.0 or float(no_timeout_terminal["result"]) != 0.0:
        raise AssertionError("no-timeout model forged a terminal Bob abort or missed the waiting state")
    counterexample = {
        "properties": [
            "Pmax=? [ F (terminal & aborted_by_alice & AliceKnowsB & !BobKnowsA) ]",
            "Pmax=? [ F (terminal & OutA_has_B & OutB_abort) ]",
        ],
        "initial_condition": {"phase": "t3", "Pass_decoy": True},
        "shortest_transition_count": len(timeout_actions),
        "required_actions": required_timeout_actions,
        "trace": timeout_trace,
        "prism_terminal_knowledge_property": prism_main,
        "prism_terminal_output_property": prism_output,
        "interpretation": "Alice's abort is nonterminal; only the explicit timeout followed by bob_abort creates terminal output failure.",
    }
    counterexample_path = COUNTEREXAMPLES / "e1_abort_terminal_shortest_counterexample_v6.json"
    write_json(counterexample_path, counterexample)
    no_timeout_witness = {
        "property": "Pmax=? [ F (!terminal & BobWaitingForCA & AliceKnowsB & !BobKnowsA & !OutB_abort) ]",
        "required_actions": no_timeout_actions,
        "trace": no_timeout_trace,
        "prism_waiting_property": no_timeout_main,
        "prism_terminal_bob_abort_property": no_timeout_terminal,
        "interpretation": "Without a timeout rule, the model records a reachable nonterminal liveness failure and does not invent Bob's abort output.",
    }
    no_timeout_path = COUNTEREXAMPLES / "e1_abort_no_timeout_liveness_witness_v6.json"
    write_json(no_timeout_path, no_timeout_witness)
    raw_prism_path = RAW / "e1_abort_terminal_mdp_v6.json"
    write_json(raw_prism_path, counterexample)

    summaries = [
        summary_row("E1", "chosen_message_pair", "chosen_message_success", 1.0, unit="probability", raw_hash=sha256_file(hash_path), notes="Analytic deterministic conclusion; script is a witness only."),
        summary_row("E1", "chosen_message_pair", "privacy_advantage", 0.5, unit="probability", raw_hash=sha256_file(hash_path), notes="Analytic deterministic conclusion."),
        summary_row("E1", "cross_session_repeated_message", "equality_identification_success", 1.0, unit="probability", raw_hash=sha256_file(hash_path)),
        summary_row("E1", "abort_terminal_pass_decoy_t3", "pmax_terminal_unfair_output", prism_main["result"], unit="probability", raw_hash=sha256_file(raw_prism_path), notes="Target requires terminal=true and aborted_by_alice=true."),
        summary_row("E1", "abort_timeout_pass_decoy_t3", "shortest_counterexample_length", len(timeout_actions), unit="transitions", raw_hash=sha256_file(counterexample_path), notes="send_cb -> alice_recovers_b -> alice_abort -> bob_waiting_for_ca -> timeout -> bob_abort"),
        summary_row("E1", "abort_terminal_pass_decoy_t3", "reachable_states", prism_main["states"], unit="states", raw_hash=sha256_file(raw_prism_path)),
        summary_row("E1", "abort_terminal_pass_decoy_t3", "transitions", prism_main["transitions"], unit="transitions", raw_hash=sha256_file(raw_prism_path)),
        summary_row("E1", "abort_no_timeout", "pmax_nonterminal_waiting_knowledge_asymmetry", no_timeout_main["result"], unit="probability", raw_hash=sha256_file(no_timeout_path), notes="Nonterminal liveness failure, not terminal fairness output."),
        summary_row("E1", "abort_no_timeout", "pmax_terminal_bob_abort", no_timeout_terminal["result"], unit="probability", raw_hash=sha256_file(no_timeout_path), notes="No timeout means Bob abort output is not fabricated."),
    ]
    write_csv(PROCESSED / "e1_summary_v6.csv", summaries)
    return summaries


if __name__ == "__main__":
    run()

