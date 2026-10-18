import numpy as np
import stim

from steane_code import noise_and_benchmarks
from hard_decision_decoder import hard_decision_decoder

# ---------------------------------------------------------------------------
# Circuit-level noise
# ---------------------------------------------------------------------------
_ANNOTATIONS = {"DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS",
                "SHIFT_COORDS", "TICK"}


def _flip_error_for_basis(name):
    """Error that flips a reset/measurement in this gate's basis."""
    return "Z_ERROR" if name.endswith("X") else "X_ERROR"


def _add_idle_noise(noisy, num_qubits, active, p_idle):
    if p_idle <= 0:
        return
    idle = sorted(set(range(num_qubits)) - active)
    if idle:
        noisy.append("DEPOLARIZE1", idle, p_idle)


def add_circuit_noise(circuit, p_idle, p_1gate, p_2gate, p_measure, p_reset,
                      num_qubits=None):
    if num_qubits is None:
        num_qubits = circuit.num_qubits
    noisy = stim.Circuit()
    active = set()

    for inst in circuit:
        # Recurse into REPEAT blocks (needed for multi-round circuits).
        if isinstance(inst, stim.CircuitRepeatBlock):
            body = add_circuit_noise(inst.body_copy(), p_idle, p_1gate, p_2gate,
                                     p_measure, p_reset, num_qubits)
            noisy.append(stim.CircuitRepeatBlock(inst.repeat_count, body))
            continue

        name = inst.name
        if name == "TICK":
            _add_idle_noise(noisy, num_qubits, active, p_idle)
            noisy.append(inst)
            active.clear()
            continue
        if name in _ANNOTATIONS:
            noisy.append(inst)
            continue

        gd = stim.gate_data(name)
        if gd.is_noisy_gate and not gd.produces_measurements:
            noisy.append(inst)          # pre-existing noise channel: keep as is
            continue

        targets = inst.targets_copy()
        qubits = [t.value for t in targets if t.is_qubit_target]
        active.update(qubits)

        if gd.produces_measurements:            # M, MX, MR, MRX, MPP, ...
            noisy.append(name, targets, p_measure)
            if gd.is_reset and p_reset > 0:     # MR / MRX: reset error after
                noisy.append(_flip_error_for_basis(name), qubits, p_reset)
        elif gd.is_reset:                       # R, RX, RY
            noisy.append(inst)
            if p_reset > 0:
                noisy.append(_flip_error_for_basis(name), qubits, p_reset)
        elif gd.is_unitary and gd.is_single_qubit_gate:   # H, S, X, ...
            noisy.append(inst)
            if p_1gate > 0:
                noisy.append("DEPOLARIZE1", qubits, p_1gate)
        elif gd.is_unitary and gd.is_two_qubit_gate:      # CX, CZ, ...
            noisy.append(inst)
            if p_2gate > 0:
                noisy.append("DEPOLARIZE2", qubits, p_2gate)
        else:
            noisy.append(inst)

    # Flush idle noise for a final moment that has no closing TICK.
    if active:
        _add_idle_noise(noisy, num_qubits, active, p_idle)
    return noisy


# ---------------------------------------------------------------------------
# Hook the hard-decision decoder up to the circuit
# ---------------------------------------------------------------------------
# ASSUMPTIONS -- edit these to match circuit_level_shor's layout:
DATA_QUBITS = list(range(9))         # circuit qubit index of data qubit 0..8
SYND_Z_DETS = list(range(0, 6))      # detectors for the 6 ZZ syndromes (HZ_PAIRS order)
SYND_X_DETS = list(range(6, 8))      # detectors for the 2 X^6 syndromes (HX_GROUPS order)
OBS_INDEX = 0


def observable_support(circuit, obs_index=OBS_INDEX):
    """(qubit, basis) pairs of measurements that make up the logical observable."""
    records, support = [], set()
    for inst in circuit.flattened():
        name = inst.name
        if name == "OBSERVABLE_INCLUDE":
            if int(inst.gate_args_copy()[0]) != obs_index:
                continue
            for t in inst.targets_copy():
                if t.is_measurement_record_target:
                    support ^= {records[len(records) + t.value]}
            continue
        if name in _ANNOTATIONS or name == "DETECTOR":
            continue
        if stim.gate_data(name).produces_measurements:
            if name == "MPP":
                raise NotImplementedError("MPP in observable not handled")
            basis = "X" if name.endswith("X") else "Z"
            for t in inst.targets_copy():
                if t.is_qubit_target:
                    records.append((t.value, basis))
    return support


def build_prediction_table(circuit):
    """Precompute predicted observable flip for all 2^8 syndromes."""
    q_to_data = {q: i for i, q in enumerate(DATA_QUBITS)}
    support = observable_support(circuit)
    n_z, n_x = len(SYND_Z_DETS), len(SYND_X_DETS)
    table = np.zeros(2 ** (n_z + n_x), dtype=np.uint8)
    for idx in range(len(table)):
        bits = [(idx >> k) & 1 for k in range(n_z + n_x)]
        cx, cz = hard_decision_decode(bits[:n_z], bits[n_z:])
        flip = 0
        for q, basis in support:
            if q not in q_to_data:
                continue
            d = q_to_data[q]
            # Z-basis readout is flipped by X corrections, X-basis by Z corrections.
            flip ^= cx[d] if basis == "Z" else cz[d]
        table[idx] = flip
    return table


def logical_error_rate(circuit, shots):
    sampler = circuit.compile_detector_sampler()
    dets, obs = sampler.sample(shots=shots, separate_observables=True)
    synd = dets[:, SYND_Z_DETS + SYND_X_DETS].astype(np.int64)
    idx = synd @ (1 << np.arange(synd.shape[1]))
    predicted = build_prediction_table(circuit)[idx]
    failures = predicted ^ obs[:, OBS_INDEX].astype(np.uint8)
    return obs[:, OBS_INDEX].mean(), failures.mean()


if __name__ == "__main__":
    level1_circuit = circuit_level_shor()   # pass your args here
    p = 1e-4
    noisy_circuit = add_circuit_noise(level1_circuit, p_idle=p, p_1gate=p,
                                      p_2gate=p, p_measure=p, p_reset=p)
    raw, decoded = logical_error_rate(noisy_circuit, shots=100_000)
    print(f"Raw observable flip rate (no decoding): {raw:.3e}")
    print(f"Logical error rate (hard decoder):      {decoded:.3e}")