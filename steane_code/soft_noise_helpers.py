# hi guys this file is just for aahnas code, ignore this
import stim


def add_idle_noise(noisy_circuit, num_qubits, active_qubits, p_idle):
    
    if p_idle <= 0:
        return

    idle_qubits = set(range(num_qubits)) - active_qubits

    if idle_qubits:
        noisy_circuit.append("DEPOLARIZE1", sorted(idle_qubits), p_idle)


def add_circuit_noise(
    noiseless_circuit,
    p_idle,
    p_1gate,
    p_2gate,
    p_measure,
    p_reset,
):
    """Returns a noisy copy of a noiseless Stim circuit"""
    noisy_circuit = stim.Circuit()
    num_qubits = noiseless_circuit.num_qubits
    active_qubits = set()

    for instruction in noiseless_circuit:
        name = instruction.name
        targets = instruction.targets_copy()
        args = instruction.gate_args_copy()

        if name == "TICK":
            add_idle_noise(
                noisy_circuit,
                num_qubits,
                active_qubits,
                p_idle,
            )
            noisy_circuit.append(instruction)
            active_qubits.clear()
            continue

        if name in ("DETECTOR", "OBSERVABLE_INCLUDE"):
            noisy_circuit.append(instruction)
            continue

        active_qubits.update(
            target.value
            for target in targets
            if target.is_qubit_target
        )

        if name == "R":
            noisy_circuit.append("R", targets)
            if p_reset > 0:
                noisy_circuit.append("X_ERROR", targets, p_reset)

        elif name == "MR":
            noisy_circuit.append("MR", targets, p_measure)
            if p_reset > 0:
                noisy_circuit.append("X_ERROR", targets, p_reset)

        elif name == "M":
            noisy_circuit.append("M", targets, p_measure)

        elif name == "H":
            noisy_circuit.append("H", targets, args)
            if p_1gate > 0:
                noisy_circuit.append(
                    "DEPOLARIZE1",
                    targets,
                    p_1gate,
                )

        elif name == "CX":
            noisy_circuit.append("CX", targets, args)
            if p_2gate > 0:
                noisy_circuit.append(
                    "DEPOLARIZE2",
                    targets,
                    p_2gate,
                )

        else:
            noisy_circuit.append(instruction)

    return noisy_circuit