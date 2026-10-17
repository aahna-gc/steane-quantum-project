import stim

"""Implement circuit-level noise"""
# Helper function: implement idle noise.
def add_idle_noise(noisy_circuit, num_qubits, active_qubits, p_idle):
    if p_idle <= 0:
        return
    
    idle_qubits = set(range(num_qubits)) - active_qubits

    if p_idle > 0 and idle_qubits:
        noisy_circuit.append("DEPOLARIZE1", idle_qubits, p_idle)

def add_circuit_noise(noiseless_circuit, p_idle, p_1gate, p_2gate, p_measure, p_reset):
    # Create a copy of the original circuit so noise can be injected.
    noisy_circuit = stim.Circuit()
    num_qubits = noiseless_circuit.num_qubits
    active_qubits = set()

    for instruction in noiseless_circuit:
        name = instruction.name
        targets = instruction.targets_copy() # extracts the qubits associated with a specific instruction
        args = instruction.gate_args_copy()

        # Idle noise
        if name == "TICK":
            add_idle_noise(noisy_circuit, num_qubits, active_qubits, p_idle)
            noisy_circuit.append(instruction)
            active_qubits.clear()
            continue
        if name in ("DETECTOR", "OBSERVABLE_INCLUDE"):
            noisy_circuit.append(instruction)
            continue
        # Track qubits participating in this moment.
        active_qubits.update(t.value for t in targets if t.is_qubit_target)

        # Reset and state preparation noise (R)
        if name == "R":
            noisy_circuit.append("R", targets)
            if p_reset > 0:
                noisy_circuit.append("X_ERROR", targets, p_reset)

        # Measurement then reset noise (MR)
        elif name == "MR":
            noisy_circuit.append("MR", targets, p_measure)
            if p_reset > 0:
                noisy_circuit.append("X_ERROR", targets, p_reset)

        # Measurement noise (M)
        elif name == "M":
            noisy_circuit.append("M", targets, p_measure)

        # Single-qubit gate noise (H)
        elif name == "H":
            noisy_circuit.append("H", targets, args)
            if p_1gate > 0:
                noisy_circuit.append("DEPOLARIZE1", targets, p_1gate)

        # Two-qubit gate noise (CX)
        elif name == "CX":
            noisy_circuit.append("CX", targets, args)
            if p_2gate > 0:
                noisy_circuit.append("DEPOLARIZE2", targets, p_2gate)
    
        # Preserve all other instructions
        else:
            noisy_circuit.append(instruction)
            
    return noisy_circuit