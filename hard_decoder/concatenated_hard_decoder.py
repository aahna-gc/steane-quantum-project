"""
Shared hard-decision decoder for the concatenated Steane circuits built by
steane_code/recursive_model.py:

    L2 = concatenate_circuit(level_1_steane(1), 1)   # [[49, 1, 9]]
    L3 = concatenate_circuit(level_1_steane(1), 2)   # [[343, 1, 27]]

Used by hard_decision_decoder_l2.py and hard_decision_decoder_l3.py; the
level-1 decoder lives in hard_decision_decoder.py, and this module reuses
its Hamming lookup table.

What the concatenated circuit looks like
----------------------------------------
concatenate_circuit replaces every qubit of level_1_steane(1) with a
7-qubit Steane block (recursively), every gate with its transversal
version, and every measurement with a transversal Z-measurement of the
whole block. So the level-1 circuit's 13 measurements -- 3 Z-stabilizer
ancillas, 3 X-stabilizer ancillas, 7 data qubits, in that order -- each
become 7**(level-1) physical bits, with the innermost block's 7 bits
adjacent (physical qubit = 7 * logical qubit + j, recursively).

How it's decoded
----------------
The same hard-decision idea as level 1, applied level by level from the
bottom up:

    Inner levels: every group of 7 measured bits is a (noisy) Hamming
        codeword. Look up its 3-bit syndrome in the SAME table as level 1,
        flip the indicated bit, and commit to its parity as the hard
        logical bit for that block. Repeat on the resulting bits until
        each of the 13 level-1 measurements is a single bit.

    Top level: exactly the level-1 decoder. The 7 decoded data bits give
        the final Z-syndrome -> stage X correction -> logical Z. The 3
        decoded X-stabilizer ancilla bits give the stage Z correction
        (which, as at level 1, can't change a Z-basis readout).

Nothing but a hard bit passes from one level to the next -- no
confidence/likelihood -- which is what makes this the hard-decision
baseline for the soft decoder.

The circuit's own DETECTORs are not used: they XOR whole blocks of raw
physical bits, which throws away the inner syndromes. The decoder works
from the raw measurement record instead. In the noiseless circuit every
decoded ancilla bit is 0 and the decoded logical value is 0, so any
deviation is an error (checked in the unit tests).
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import stim

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "steane_code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from level1_steane import level_1_steane  # noqa: E402
from recursive_model import concatenate_circuit  # noqa: E402
from noise import add_circuit_noise  # noqa: E402
from hard_decision_decoder import (  # noqa: E402
    NUM_QUBITS,
    STABILIZERS,
    _CORRECTION_FLIPS_LOGICAL,
    _CORRECTION_QUBIT,
    _PACK,
)

# The 13 measurements of level_1_steane(1), in record order.
Z_ANCILLA_SLOTS = slice(0, 3)
X_ANCILLA_SLOTS = slice(3, 6)
DATA_SLOTS = slice(6, 13)
NUM_TOP_MEASUREMENTS = 13

# H[i, j] = 1 if data qubit j is in stabilizer i.
_PARITY_CHECK = np.array(
    [[int(q in stab) for q in range(NUM_QUBITS)] for stab in STABILIZERS],
    dtype=np.int64,
)

NOISE_LEVELS = [1e-3, 3e-4, 1e-4, 3e-5, 1e-5]
SHOTS = 100_000


def build_circuit(level: int) -> stim.Circuit:
    """Project level -> circuit. Level 1 is level_1_steane(1) itself;
    level L >= 2 is concatenate_circuit(level_1_steane(1), L - 1)."""
    return concatenate_circuit(level_1_steane(1), level - 1)


def _block_syndrome_index(bits: np.ndarray) -> np.ndarray:
    """(..., 7) bits -> (...) packed syndrome s0 + 2*s1 + 4*s2."""
    synd = (bits.astype(np.int64) @ _PARITY_CHECK.T) % 2
    return synd @ _PACK


def decode_block(bits: np.ndarray) -> np.ndarray:
    """One hard-decision step: (..., 7) noisy codeword bits -> (...) logical
    bit, after the Hamming-lookup correction."""
    raw_parity = bits.sum(axis=-1) % 2
    flip = _CORRECTION_FLIPS_LOGICAL[_block_syndrome_index(bits)]
    return (raw_parity ^ flip).astype(np.uint8)


def decode_to_top_level(meas: np.ndarray, level: int) -> np.ndarray:
    """Raw measurement record (shots, 13 * 7**(level-1)) -> hard logical bits
    (shots, 13) for the 13 measurements of the level-1 circuit."""
    bits = np.asarray(meas, dtype=np.uint8).reshape(
        meas.shape[0], NUM_TOP_MEASUREMENTS, 7 ** (level - 1)
    )
    for _ in range(level - 1):
        bits = decode_block(bits.reshape(*bits.shape[:-1], -1, 7))
    return bits.reshape(meas.shape[0], NUM_TOP_MEASUREMENTS)


def decode_shots(meas: np.ndarray, level: int) -> Dict[str, np.ndarray]:
    """Full decoder. Returns per-shot:
        decoded_logical: hard-decision logical Z (0 = correct)
        raw_logical:     parity of all physical data bits (no decoding;
                         equals the circuit's OBSERVABLE 0)
        x_correction_qubit / z_correction_qubit: top-level corrections
    """
    meas = np.asarray(meas, dtype=np.uint8)
    top = decode_to_top_level(meas, level)
    data = top[:, DATA_SLOTS]
    x_anc = top[:, X_ANCILLA_SLOTS].astype(np.int64)

    z_idx = _block_syndrome_index(data)
    n_data_bits = 7 * 7 ** (level - 1)
    return {
        "decoded_logical": (data.sum(axis=1) % 2) ^ _CORRECTION_FLIPS_LOGICAL[z_idx],
        "raw_logical": meas[:, -n_data_bits:].sum(axis=1) % 2,
        "x_correction_qubit": _CORRECTION_QUBIT[z_idx],
        "z_correction_qubit": _CORRECTION_QUBIT[x_anc @ _PACK],
    }


def run_benchmark(
    level: int,
    output_csv: Path,
    noise_levels: Sequence[float] = NOISE_LEVELS,
    shots: int = SHOTS,
) -> List[dict]:
    circuit = build_circuit(level)
    print(f"Level-{level} Steane hard-decision decoding benchmark")
    print(f"qubits={circuit.num_qubits}, measurements={circuit.num_measurements}, "
          f"shots per p={shots}\n")

    results = []
    for p in noise_levels:
        noisy = add_circuit_noise(
            circuit, p_idle=p, p_1gate=p, p_2gate=p, p_measure=p, p_reset=p
        )
        meas = noisy.compile_sampler().sample(shots=shots)

        start = time.perf_counter()
        out = decode_shots(meas, level)
        runtime = time.perf_counter() - start

        raw_failures = int(out["raw_logical"].sum())
        decoded_failures = int(out["decoded_logical"].sum())
        row = {
            "physical_error_rate": p,
            "level": level,
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
# Unit tests (shared by the L2 and L3 entry points).
# ---------------------------------------------------------------------------

def _final_data_qubits(circuit: stim.Circuit) -> List[int]:
    last_m = [inst for inst in circuit if inst.name == "M"][-1]
    return [t.value for t in last_m.targets_copy()]


def _with_x_before_readout(circuit: stim.Circuit, qubits: Sequence[int]) -> stim.Circuit:
    """Copy of `circuit` with deterministic X errors just before the final M."""
    insts = list(circuit)
    last_m = max(i for i, inst in enumerate(insts) if inst.name == "M")
    out = stim.Circuit()
    for i, inst in enumerate(insts):
        if i == last_m:
            out.append("X_ERROR", list(qubits), 1.0)
        out.append(inst)
    return out


def _decode_one(circuit: stim.Circuit, level: int) -> Dict[str, np.ndarray]:
    meas = circuit.compile_sampler().sample(shots=8)
    return decode_shots(meas, level)


def run_unit_tests(level: int) -> None:
    circuit = build_circuit(level)
    data = _final_data_qubits(circuit)
    block = 7 ** (level - 1)  # physical qubits per top-level data qubit
    print(f"Level-{level} circuit: {circuit.num_qubits} qubits, "
          f"{circuit.num_measurements} measurements")

    # Noiseless: every decoded ancilla bit and the logical must be 0.
    meas = circuit.compile_sampler().sample(shots=256)
    top = decode_to_top_level(meas, level)
    assert not top[:, :6].any(), "noiseless ancilla bits should all be 0"
    out = decode_shots(meas, level)
    assert not out["decoded_logical"].any() and not out["raw_logical"].any()
    assert (out["x_correction_qubit"] == -1).all()
    print("  PASS: noiseless -> ancillas 0, logical 0, no correction")

    # Every single physical X before readout is corrected.
    for q in data:
        out = _decode_one(_with_x_before_readout(circuit, [q]), level)
        assert out["raw_logical"].all(), f"X on {q} should flip the raw readout"
        assert not out["decoded_logical"].any(), f"X on {q} not corrected"
    print(f"  PASS: single X before readout on each of {len(data)} data qubits -> corrected")

    # Two X's in one innermost block beat that block's lookup (it outputs
    # the wrong bit), but the level above sees a single wrong bit and fixes it.
    two_in_block = [data[0], data[1]]
    out = _decode_one(_with_x_before_readout(circuit, two_in_block), level)
    assert not out["decoded_logical"].any()
    print("  PASS: 2 X's in one innermost block (inner lookup fails) -> "
          "fixed by the level above")

    # Same thing in 2 different top-level data blocks: two wrong bits at the
    # top level, which the top-level lookup can't fix -> logical failure.
    # This is the expected distance-limited failure, not a decoder bug.
    if level == 2:
        bad = [data[0], data[1], data[block], data[block + 1]]
        out = _decode_one(_with_x_before_readout(circuit, bad), level)
        assert out["decoded_logical"].all()
        print("  PASS: inner failures in 2 different top-level blocks -> "
              "logical failure (expected, weight-4 < d=9 but badly placed)")

    print("\nALL UNIT TESTS PASSED.")


def main(level: int, output_csv: Path) -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        run_benchmark(level, output_csv)
    else:
        run_unit_tests(level)
