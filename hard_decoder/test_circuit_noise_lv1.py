"""Tests for circuit_noise_level1.py.

Run with:  pytest -v test_circuit_noise_lv1.py

Sections:
  1. add_circuit_noise   -- tiny hand-written circuits, exact noise placement
  2. observable_support  -- measurement-record bookkeeping
  3. decoder hookup      -- synthetic circuits with forced (p=1) errors, so the
                            expected decoded outcome is known exactly
  4. real Shor circuit   -- skipped automatically if circuit_level_shor()
                            can't be built with no arguments
"""

import numpy as np
import pytest
import stim

import circuit_noise_level1 as cn
from shor9_code import BLOCKS

P1, P2, PM, PR, PI = 0.01, 0.03, 0.04, 0.02, 0.1   # distinct values on purpose


def ops(circuit):
    """Flat list of (name, args, targets) for top-level instructions."""
    out = []
    for inst in circuit:
        if isinstance(inst, stim.CircuitRepeatBlock):
            out.append(("REPEAT", (inst.repeat_count,), ()))
        else:
            out.append((inst.name,
                        tuple(inst.gate_args_copy()),
                        tuple(t.value for t in inst.targets_copy())))
    return out


def noisy(text, p_idle=0, p_1gate=0, p_2gate=0, p_measure=0, p_reset=0):
    return cn.add_circuit_noise(stim.Circuit(text), p_idle=p_idle,
                                p_1gate=p_1gate, p_2gate=p_2gate,
                                p_measure=p_measure, p_reset=p_reset)


# ---------------------------------------------------------------------------
# 1. add_circuit_noise
# ---------------------------------------------------------------------------

def test_single_qubit_gate_gets_depolarize1():
    assert ops(noisy("H 0\nTICK", p_1gate=P1)) == [
        ("H", (), (0,)),
        ("DEPOLARIZE1", (P1,), (0,)),
        ("TICK", (), ()),
    ]


@pytest.mark.parametrize("gate", ["CX", "CZ"])
def test_two_qubit_gate_gets_depolarize2(gate):
    assert ops(noisy(f"{gate} 0 1", p_2gate=P2)) == [
        (gate, (), (0, 1)),
        ("DEPOLARIZE2", (P2,), (0, 1)),
    ]


def test_measurement_gets_flip_probability():
    assert ops(noisy("M 0 1", p_measure=PM)) == [("M", (PM,), (0, 1))]
    assert ops(noisy("MX 0", p_measure=PM)) == [("MX", (PM,), (0,))]


def test_reset_error_matches_basis():
    assert ops(noisy("R 0\nRX 1", p_reset=PR)) == [
        ("R", (), (0,)),
        ("X_ERROR", (PR,), (0,)),
        ("RX", (), (1,)),
        ("Z_ERROR", (PR,), (1,)),
    ]


def test_measure_reset_gets_both_errors():
    assert ops(noisy("MR 0", p_measure=PM, p_reset=PR)) == [
        ("MR", (PM,), (0,)),
        ("X_ERROR", (PR,), (0,)),
    ]
    assert ops(noisy("MRX 0", p_measure=PM, p_reset=PR)) == [
        ("MRX", (PM,), (0,)),
        ("Z_ERROR", (PR,), (0,)),
    ]


def test_idle_noise_only_hits_idle_qubits():
    assert ops(noisy("R 0 1 2\nTICK\nH 0\nTICK", p_idle=PI)) == [
        ("R", (), (0, 1, 2)),
        ("TICK", (), ()),                      # all 3 active -> no idle noise
        ("H", (), (0,)),
        ("DEPOLARIZE1", (PI,), (1, 2)),        # qubits 1, 2 sat idle
        ("TICK", (), ()),
    ]


def test_final_moment_without_tick_gets_idle_noise():
    assert ops(noisy("R 0 1\nTICK\nH 0", p_idle=PI)) == [
        ("R", (), (0, 1)),
        ("TICK", (), ()),
        ("H", (), (0,)),
        ("DEPOLARIZE1", (PI,), (1,)),
    ]


def test_zero_idle_probability_adds_nothing():
    assert ops(noisy("R 0 1\nTICK\nH 0\nTICK")) == [
        ("R", (), (0, 1)), ("TICK", (), ()), ("H", (), (0,)), ("TICK", (), ()),
    ]


def test_repeat_block_is_noised_with_full_qubit_count():
    c = noisy("R 0 1\nTICK\nREPEAT 3 {\n H 0\n TICK\n}", p_1gate=P1, p_idle=PI)
    *_, block = list(c)
    assert isinstance(block, stim.CircuitRepeatBlock)
    assert block.repeat_count == 3
    # The body alone only touches qubit 0; qubit 1 must still get idle noise.
    assert ops(block.body_copy()) == [
        ("H", (), (0,)),
        ("DEPOLARIZE1", (P1,), (0,)),
        ("DEPOLARIZE1", (PI,), (1,)),
        ("TICK", (), ()),
    ]


def test_existing_noise_and_annotations_preserved():
    text = ("QUBIT_COORDS(0, 0) 0\nR 0\nX_ERROR(0.3) 0\nM 0\n"
            "DETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]")
    c = noisy(text)
    names = [o[0] for o in ops(c)]
    assert names.count("X_ERROR") == 1
    assert c.num_detectors == 1 and c.num_observables == 1
    assert "QUBIT_COORDS" in names


def test_zero_noise_gives_no_detection_events():
    text = ("R 0 1\nTICK\nCX 0 1\nTICK\nM 0 1\n"
            "DETECTOR rec[-1]\nDETECTOR rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]")
    c = noisy(text)
    dets, obs = c.compile_detector_sampler().sample(500, separate_observables=True)
    assert not dets.any() and not obs.any()


# ---------------------------------------------------------------------------
# 2. observable_support
# ---------------------------------------------------------------------------

def test_observable_support_tracks_qubit_and_basis():
    c = stim.Circuit("M 0 1\nMX 2\nOBSERVABLE_INCLUDE(0) rec[-1] rec[-3]")
    assert cn.observable_support(c) == {(2, "X"), (0, "Z")}


def test_observable_support_double_inclusion_cancels():
    c = stim.Circuit("M 0\nOBSERVABLE_INCLUDE(0) rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]")
    assert cn.observable_support(c) == set()


def test_observable_support_ignores_other_observables():
    c = stim.Circuit("M 0\nOBSERVABLE_INCLUDE(1) rec[-1]")
    assert cn.observable_support(c, obs_index=0) == set()


def test_observable_support_ignores_detectors_and_noise():
    c = stim.Circuit("X_ERROR(0.1) 0\nM 0 1\nDETECTOR rec[-2]\n"
                     "OBSERVABLE_INCLUDE(0) rec[-1]")
    assert cn.observable_support(c) == {(1, "Z")}


# ---------------------------------------------------------------------------
# 3. Decoder hookup (synthetic circuits, deterministic errors)
#
# Layout: data qubits 0..8 (matches DATA_QUBITS), syndrome qubits 9..16.
# Detectors 0..7 come from syndrome qubits 9..16 in order, matching
# SYND_Z_DETS = 0..5 and SYND_X_DETS = 6..7. An X_ERROR(1) on a syndrome qubit
# forces that detector to fire on every shot.
# ---------------------------------------------------------------------------

SYND_QUBITS = list(range(9, 17))


def synthetic_circuit(fired_dets, data_error=None, data_basis="Z"):
    d = BLOCKS[0][0]                      # qubit the decoder corrects for "symbol 0"
    lines = []
    if data_basis == "X":
        lines.append("RX " + " ".join(map(str, range(9))))
    if data_error:
        lines.append(f"{data_error}(1) {d}")
    for k in fired_dets:
        lines.append(f"X_ERROR(1) {SYND_QUBITS[k]}")
    lines.append("M " + " ".join(map(str, SYND_QUBITS)))
    for i in range(8):
        lines.append(f"DETECTOR rec[{-8 + i}]")
    lines.append(("MX " if data_basis == "X" else "M ") + " ".join(map(str, range(9))))
    lines.append(f"OBSERVABLE_INCLUDE(0) rec[{-9 + d}]")
    return stim.Circuit("\n".join(lines))


def test_prediction_table_shape_and_trivial_syndrome():
    table = cn.build_prediction_table(synthetic_circuit([]))
    assert table.shape == (256,)
    assert table[0] == 0


def test_no_errors_no_failures():
    raw, decoded = cn.logical_error_rate(synthetic_circuit([]), shots=200)
    assert raw == 0 and decoded == 0


def test_level1_corrects_x_error_it_detects():
    # Data X error + its matching syndrome (1,0) in block 0 -> decoder fixes it.
    raw, decoded = cn.logical_error_rate(
        synthetic_circuit([0], data_error="X_ERROR"), shots=200)
    assert raw == 1 and decoded == 0


def test_level1_spurious_syndrome_causes_failure():
    # Syndrome fires but data is clean -> decoder's correction itself flips
    # the observable (models a syndrome-measurement error).
    raw, decoded = cn.logical_error_rate(synthetic_circuit([0]), shots=200)
    assert raw == 0 and decoded == 1


def test_level2_corrects_z_error_in_x_basis():
    raw, decoded = cn.logical_error_rate(
        synthetic_circuit([6], data_error="Z_ERROR", data_basis="X"), shots=200)
    assert raw == 1 and decoded == 0


def test_x_correction_ignored_for_x_basis_readout():
    # Z-type syndrome fires; X correction doesn't affect an MX readout.
    raw, decoded = cn.logical_error_rate(
        synthetic_circuit([0], data_basis="X"), shots=200)
    assert raw == 0 and decoded == 0


# ---------------------------------------------------------------------------
# 4. Real Shor circuit (auto-skipped if it can't be built with no args)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def shor_circuit():
    try:
        return cn.circuit_level_shor()
    except Exception as e:  # signature needs args, or not implemented yet
        pytest.skip(f"circuit_level_shor() not buildable with no args: {e!r}")


def test_shor_layout_matches_assumptions(shor_circuit):
    assert shor_circuit.num_detectors >= len(cn.SYND_Z_DETS) + len(cn.SYND_X_DETS)
    assert shor_circuit.num_observables >= 1
    assert max(cn.DATA_QUBITS) < shor_circuit.num_qubits


def test_shor_detectors_are_deterministic(shor_circuit):
    # Raises if any detector/observable is non-deterministic without noise.
    shor_circuit.detector_error_model()


def test_shor_noiseless_has_zero_logical_error(shor_circuit):
    raw, decoded = cn.logical_error_rate(shor_circuit, shots=2_000)
    assert raw == 0 and decoded == 0


def test_shor_noise_preserves_structure(shor_circuit):
    c = cn.add_circuit_noise(shor_circuit, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3)
    assert c.num_detectors == shor_circuit.num_detectors
    assert c.num_observables == shor_circuit.num_observables
    assert c.num_measurements == shor_circuit.num_measurements
    c.detector_error_model()   # still a valid, deterministic-detector circuit
