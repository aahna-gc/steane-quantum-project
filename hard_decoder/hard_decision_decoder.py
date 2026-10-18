"""
Hard-decision decoder for the [[7,1,3]] Steane code built in
steane_code/level1_steane.py.

This is the Steane-code counterpart of the earlier Shor-9 hard-decision
decoder. The decoding is done the same way: two independent stages, each
of which only ever sees a HARD 3-bit syndrome and commits to a single
guess, with no confidence/likelihood carried between stages, rounds, or
qubits.

    Stage X (bit-flip correction): the 3 Z-stabilizer syndrome bits are
        looked up in the Hamming(7,4) syndrome table to guess which single
        data qubit (if any) suffered an X error.

    Stage Z (phase-flip correction): the 3 X-stabilizer syndrome bits are
        looked up in the SAME table to guess which single data qubit (if
        any) suffered a Z error.

(The Shor version called these "level 1" and "level 2"; they're renamed
here because "level" in this project means concatenation level --
level_1_steane, benchmarks_l2, benchmarks_l3.)
The Steane code is CSS and self-dual: its X- and Z-stabilizers have the
same supports (STABILIZERS below, identical to level1_steane.py). So, just
as the Shor decoder reused `_REP3_LOOKUP` for both of its levels, one
lookup table `_HAMMING_LOOKUP` is used for both stages. Unlike Shor-9, every
nonzero syndrome points at exactly one qubit (each of the 7 qubits has a
distinct syndrome column), so there is no block-level vote: the lookup is
the full decoder.

Running on the circuit
----------------------
level_1_steane(rounds) prepares logical |0>, runs `rounds` rounds of
syndrome extraction, measures all 7 data qubits in Z, and declares
observable 0 = parity of those 7 bits (= logical Z). Its detectors are,
per round, [X2, X1, X0, Z2, Z1, Z0] (stabilizer index 2 first, because
the DETECTOR instructions walk rec(-1), rec(-2), ...).

The circuit's own detectors never look at the final data readout, so a
single X error after the last syndrome round would go uncorrected.
`build_decoding_circuit` therefore appends 3 extra detectors: the
Z-stabilizer parities of the measured data bits. That gives a final
Z-syndrome taken from the very bits the observable is computed from, and
stage X decodes it. Stage Z still runs (on the last mid-circuit X
syndrome), but a Z correction can't change a Z-basis readout, so it
doesn't affect the logical error rate of this Z-memory experiment.

Usage:
    python hard_decoder/hard_decision_decoder.py          # unit tests
    python hard_decoder/hard_decision_decoder.py bench    # Monte Carlo
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import stim

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "steane_code"))

from level1_steane import level_1_steane  # noqa: E402
from noise import add_circuit_noise  # noqa: E402


NUM_QUBITS = 7
NUM_STABILIZERS = 3

# Data-qubit supports of the 3 stabilizers, in level1_steane.py's order.
# Used for both the Z-stabilizers and the X-stabilizers (self-dual code).
STABILIZERS = (
    (3, 4, 5, 6),
    (1, 2, 5, 6),
    (0, 2, 4, 6),
)

# Detectors emitted per syndrome round by level1_steane.py.
DETECTORS_PER_ROUND = 6


def _syndrome_of_qubit(qubit: int) -> Tuple[int, ...]:
    """Syndrome bits (s0, s1, s2) triggered by a single error on `qubit`."""
    return tuple(int(qubit in stab) for stab in STABILIZERS)


# Syndrome (s0, s1, s2) -> index of the flipped data qubit, or None if the
# syndrome says "no error". Built from STABILIZERS so it can't drift out of
# sync with the circuit. With this stabilizer ordering it comes out as:
#   (0,0,0) -> None   (0,0,1) -> 0   (0,1,0) -> 1   (0,1,1) -> 2
#   (1,0,0) -> 3      (1,0,1) -> 4   (1,1,0) -> 5   (1,1,1) -> 6
# i.e. the syndrome, read as a binary number s0 s1 s2, is (qubit + 1).
_HAMMING_LOOKUP: Dict[Tuple[int, ...], Optional[int]] = {
    (0, 0, 0): None,
    **{_syndrome_of_qubit(q): q for q in range(NUM_QUBITS)},
}
assert len(_HAMMING_LOOKUP) == 2 ** NUM_STABILIZERS


def _decode_single_error(syndrome: Sequence[int]) -> List[int]:
    """3-bit syndrome -> length-7 correction vector (at most one 1)."""
    correction = [0] * NUM_QUBITS
    flipped = _HAMMING_LOOKUP[tuple(int(s) for s in syndrome)]
    if flipped is not None:
        correction[flipped] = 1
    return correction


def decode_bitflip(synd_z: Sequence[int]) -> List[int]:
    """Stage X: 3-bit Z-stabilizer syndrome -> length-7 X-correction vector."""
    return _decode_single_error(synd_z)


def decode_phaseflip(synd_x: Sequence[int]) -> List[int]:
    """Stage Z: 3-bit X-stabilizer syndrome -> length-7 Z-correction vector.
    Same table as stage X -- the X-stabilizers have the same supports."""
    return _decode_single_error(synd_x)


def hard_decision_decode(
    synd_z: Sequence[int], synd_x: Sequence[int], p: float = None
) -> Tuple[List[int], List[int]]:
    """Full hard-decision decoder. `p` is accepted (and ignored) so this
    has the same call signature as a likelihood-based decoder and the two
    can be swapped in the same benchmark."""
    return decode_bitflip(synd_z), decode_phaseflip(synd_x)


# ---------------------------------------------------------------------------
# Code-capacity helpers (syndromes / logicals of a bare error vector).
# ---------------------------------------------------------------------------

def syndrome(error: Sequence[int]) -> List[int]:
    """Syndrome of an X (or Z) error vector, s_i = parity over STABILIZERS[i]."""
    return [sum(error[q] for q in stab) % 2 for stab in STABILIZERS]


def logical_flip(error: Sequence[int]) -> int:
    """Whether an X (or Z) error vector flips logical Z (or X).
    Z_bar = Z^7 and X_bar = X^7, so this is the total parity."""
    return sum(error) % 2


# ---------------------------------------------------------------------------
# Running the decoder on level1_steane.py circuits.
# ---------------------------------------------------------------------------

def build_decoding_circuit(rounds: int = 1) -> stim.Circuit:
    """level_1_steane(rounds) plus 3 detectors on the final data readout:
    detector k = parity of the measured data bits in STABILIZERS[k]."""
    circuit = level_1_steane(rounds)
    # The final M of qubits 0..6 is the last 7 records: qubit q is rec(q - 7).
    for stab in STABILIZERS:
        circuit.append(
            "DETECTOR", [stim.target_rec(q - NUM_QUBITS) for q in stab]
        )
    return circuit


def _round_syndromes(dets: np.ndarray, rounds: int) -> Tuple[np.ndarray, np.ndarray]:
    """Convert mid-circuit detectors to absolute per-round syndromes.

    dets: (shots, num_detectors) bool array.
    Returns (synd_x, synd_z), each (shots, rounds, 3) in stabilizer order
    s0, s1, s2. Rounds after the first are difference detectors, so the
    absolute syndrome is their running XOR."""
    shots = dets.shape[0]
    per_round = dets[:, : rounds * DETECTORS_PER_ROUND].reshape(
        shots, rounds, DETECTORS_PER_ROUND
    )
    # [X2, X1, X0, Z2, Z1, Z0] -> reverse each half into s0, s1, s2 order.
    diff_x = per_round[:, :, 2::-1]
    diff_z = per_round[:, :, 5:2:-1]
    synd_x = np.bitwise_xor.accumulate(diff_x, axis=1)
    synd_z = np.bitwise_xor.accumulate(diff_z, axis=1)
    return synd_x, synd_z


def _vectorized_tables() -> Tuple[np.ndarray, np.ndarray]:
    """Lookup tables indexed by the syndrome packed as s0 + 2*s1 + 4*s2:
    which qubit gets corrected (-1 = none), and whether that correction
    flips the logical observable."""
    qubit = np.full(2 ** NUM_STABILIZERS, -1, dtype=np.int64)
    flips = np.zeros(2 ** NUM_STABILIZERS, dtype=bool)
    for synd, q in _HAMMING_LOOKUP.items():
        idx = synd[0] + 2 * synd[1] + 4 * synd[2]
        correction = _decode_single_error(synd)
        qubit[idx] = -1 if q is None else q
        flips[idx] = bool(logical_flip(correction))
    return qubit, flips


_PACK = np.array([1, 2, 4], dtype=np.int64)
_CORRECTION_QUBIT, _CORRECTION_FLIPS_LOGICAL = _vectorized_tables()


def decode_shots(dets: np.ndarray, rounds: int) -> Dict[str, np.ndarray]:
    """Decode a batch of shots sampled from build_decoding_circuit(rounds).

    Returns:
        predicted_flip:  (shots,) predicted flip of observable 0
        x_correction_qubit / z_correction_qubit: (shots,) qubit index, -1 = none
    """
    dets = np.asarray(dets, dtype=bool)
    synd_x, _ = _round_syndromes(dets, rounds)

    final_synd_z = dets[:, rounds * DETECTORS_PER_ROUND:].astype(np.int64)
    last_synd_x = synd_x[:, -1, :].astype(np.int64)

    z_idx = final_synd_z @ _PACK
    x_idx = last_synd_x @ _PACK
    return {
        "predicted_flip": _CORRECTION_FLIPS_LOGICAL[z_idx],
        "x_correction_qubit": _CORRECTION_QUBIT[z_idx],
        "z_correction_qubit": _CORRECTION_QUBIT[x_idx],
    }


def decode_one_shot(detectors: Sequence[int], rounds: int) -> Tuple[List[int], List[int]]:
    """Single-shot version: returns (x_correction, z_correction) vectors
    via hard_decision_decode, for inspecting individual shots."""
    dets = np.asarray(detectors, dtype=bool)[None, :]
    synd_x, _ = _round_syndromes(dets, rounds)
    final_synd_z = dets[0, rounds * DETECTORS_PER_ROUND:].astype(int).tolist()
    return hard_decision_decode(final_synd_z, synd_x[0, -1].astype(int).tolist())


# ---------------------------------------------------------------------------
# Monte Carlo benchmark (same noise model / p sweep as benchmarks_l1.py).
# ---------------------------------------------------------------------------

NOISE_LEVELS = [1e-3, 3e-4, 1e-4, 3e-5, 1e-5]
SHOTS = 100_000
ROUNDS = 1
OUTPUT_CSV = Path(__file__).resolve().parent / "level1_hard_decoded_results.csv"


def run_benchmark(
    noise_levels: Sequence[float] = NOISE_LEVELS,
    shots: int = SHOTS,
    rounds: int = ROUNDS,
    output_csv: Path = OUTPUT_CSV,
) -> List[dict]:
    circuit = build_decoding_circuit(rounds)
    print("Level-1 Steane hard-decision decoding benchmark")
    print(f"rounds={rounds}, shots per p={shots}, "
          f"detectors={circuit.num_detectors}\n")

    results = []
    for p in noise_levels:
        noisy = add_circuit_noise(
            circuit, p_idle=p, p_1gate=p, p_2gate=p, p_measure=p, p_reset=p
        )
        dets, obs = noisy.compile_detector_sampler().sample(
            shots=shots, separate_observables=True
        )
        obs = obs[:, 0]

        start = time.perf_counter()
        predicted = decode_shots(dets, rounds)["predicted_flip"]
        runtime = time.perf_counter() - start

        raw_failures = int(obs.sum())
        decoded_failures = int((obs ^ predicted).sum())
        row = {
            "physical_error_rate": p,
            "rounds": rounds,
            "shots": shots,
            "raw_logical_failures": raw_failures,
            "raw_logical_error_rate": raw_failures / shots,
            "decoded_logical_failures": decoded_failures,
            "decoded_logical_error_rate": decoded_failures / shots,
            "decoder_runtime_seconds": runtime,
            "decoder_microseconds_per_shot": runtime / shots * 1e6,
        }
        results.append(row)
        print(f"p={p:.1e}  raw={row['raw_logical_error_rate']:.3e}  "
              f"decoded={row['decoded_logical_error_rate']:.3e}  "
              f"({decoded_failures}/{shots})")

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to: {output_csv}")
    return results


# ---------------------------------------------------------------------------
# Hand-checkable unit tests.
# ---------------------------------------------------------------------------

def _xor(a: Sequence[int], b: Sequence[int]) -> List[int]:
    return [x ^ y for x, y in zip(a, b)]


def _unit(*qubits: int) -> List[int]:
    e = [0] * NUM_QUBITS
    for q in qubits:
        e[q] ^= 1
    return e


def _run_unit_tests() -> None:
    def check_case(stage, decode, name, error, expect_logical_flip):
        s = syndrome(error)
        c = decode(s)
        residual = _xor(error, c)
        assert syndrome(residual) == [0, 0, 0], (
            f"[{name}] residual {residual} is not a codeword")
        got = logical_flip(residual)
        assert got == expect_logical_flip, (
            f"[{name}] expected logical_flip={expect_logical_flip}, got {got} "
            f"(error={error}, syndrome={s}, correction={c}, residual={residual})"
        )
        print(f"  PASS: {name} -> logical_{stage}_flip={got}")

    for stage, label, decode in (
        ("z", "Stage X (bit-flip, Z-stabilizer syndrome)", decode_bitflip),
        ("x", "Stage Z (phase-flip, X-stabilizer syndrome)", decode_phaseflip),
    ):
        print(f"{label}:")
        check_case(stage, decode, "no errors", _unit(), 0)
        for q in range(NUM_QUBITS):
            check_case(stage, decode, f"single error on qubit {q}", _unit(q), 0)
        check_case(
            stage, decode,
            "double error on qubits 0,1 -- syndrome looks like qubit 2, "
            "decoder completes a weight-3 logical, FAILS",
            _unit(0, 1), 1,
        )
        check_case(
            stage, decode,
            "hook-type pair on qubits 5,6 -- looks like qubit 0, FAILS",
            _unit(5, 6), 1,
        )
        check_case(
            stage, decode,
            "weight-3 logical on qubits 0,1,2 -- zero syndrome, undetectable",
            _unit(0, 1, 2), 1,
        )
        check_case(
            stage, decode,
            "weight-4 stabilizer on qubits 3,4,5,6 -- harmless",
            _unit(3, 4, 5, 6), 0,
        )
        print()

    print("Circuit integration (level_1_steane):")
    for rounds in (1, 3):
        circuit = build_decoding_circuit(rounds)
        # Noiseless: every detector and the observable must be deterministic 0.
        dets, obs = circuit.compile_detector_sampler().sample(
            shots=16, separate_observables=True)
        assert not dets.any() and not obs.any()
        out = decode_shots(dets, rounds)
        assert not out["predicted_flip"].any()
        print(f"  PASS: rounds={rounds} noiseless -> no detections, no correction")

        # A single X on any data qubit just before readout must be corrected.
        n_mid = rounds * DETECTORS_PER_ROUND
        for q in range(NUM_QUBITS):
            noisy = stim.Circuit()
            last_m = len(circuit) - 1 - NUM_STABILIZERS - 1  # final "M" instruction
            for i, inst in enumerate(circuit):
                if i == last_m:
                    noisy.append("X_ERROR", [q], 1.0)
                noisy.append(inst)
            dets, obs = noisy.compile_detector_sampler().sample(
                shots=1, separate_observables=True)
            assert obs[0, 0], "single X should flip the raw observable"
            assert not dets[0, :n_mid].any()
            pred = decode_shots(dets, rounds)["predicted_flip"]
            assert pred[0] == obs[0, 0], f"rounds={rounds}: X on qubit {q} not corrected"
            x_corr, _ = decode_one_shot(dets[0], rounds)
            assert x_corr == _unit(q)
        print(f"  PASS: rounds={rounds} single X before readout on each qubit -> corrected")

        # A single Z on a data qubit before the syndrome round is seen by
        # stage Z and located exactly.
        text = str(circuit)
        marker = "CX 0 6 1 6 3 6\nTICK\n"  # end of the encoder
        assert marker in text
        for q in range(NUM_QUBITS):
            noisy = stim.Circuit(text.replace(marker, marker + f"Z_ERROR(1) {q}\n", 1))
            dets, _ = noisy.compile_detector_sampler().sample(
                shots=1, separate_observables=True)
            _, z_corr = decode_one_shot(dets[0], rounds)
            assert z_corr == _unit(q), f"Z on qubit {q}: got {z_corr}"
        print(f"  PASS: rounds={rounds} single Z after encoding on each qubit -> located")

    print("\nALL UNIT TESTS PASSED.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        run_benchmark()
    else:
        _run_unit_tests()
