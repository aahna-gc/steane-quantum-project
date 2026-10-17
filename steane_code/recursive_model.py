import stim
from level1_steane import level_1_steane

"""Prepare the CNOT / H gates required for setting up the Steane code."""
steane_cx_layers = [
    [(0, 2), (1, 2)],
    [(0, 4), (3, 4)],
    [(1, 5), (3, 5)],
    [(0, 6), (1, 6), (3, 6)]
]
steane_h_layers = [0, 1, 3]

def logical_qubits(circuit, block_bases): # block_bases is a list of the starting qubit indices for each 7-qubit Steane block. 7 blocks per logical qubit.
    """Prepare the |0_L> state in each seven-qubit Steane block."""
    # Reset all physical qubits in the blocks.
    circuit.append("R", [base+j for base in block_bases for j in range(7)])
    circuit.append("TICK")

    # Hadamard layer
    circuit.append("H", [base+j for base in block_bases for j in steane_h_layers])
    circuit.append("TICK")

    # CNOT encoding layers
    for layer in steane_cx_layers:
        for control, target in layer:
            targets = [q for base in block_bases for q in (base+control, base+target)]
            circuit.append("CX", targets)
        circuit.append("TICK")

def concatenate_block(circuit):
    concatenated_circuit = stim.Circuit()

    # Record in the input measurement index in relation to its output physical record indices for detectors.
    record_map = []
    input_record_count = 0
    output_record_count = 0

    for instruction in circuit:
        name = instruction.name
        targets = instruction.targets_copy()
        args = instruction.gate_args_copy()

        if name == "TICK":
            concatenated_circuit.append("TICK")
            continue

        # Expand detector and observables.
        if name in ("DETECTOR", "OBSERVABLE_INCLUDE"):
            expanded = []
            for target in targets:
                input_index = input_record_count + target.value
                for physical_index in record_map[input_index]:
                    expanded.append(stim.target_rec(physical_index - output_record_count))
            concatenated_circuit.append(name, expanded, args)
            continue

        # Prepare the |0_L> state (reset)
        if name == "R":
            bases = [t.value*7 for t in targets]
            logical_qubits(concatenated_circuit, bases)
            continue

        # Transversal logical H
        if name == "H":
            physical_targets = [t.value * 7 + j for t in targets for j in range(7)]
            concatenated_circuit.append("H", physical_targets, args)
            continue

        # Transversal logical CNOT
        if name == "CX":
            physical_targets = []
            for i in range(0, len(targets), 2):
                # Set up controls and targets
                control = targets[i].value * 7
                target = targets[i+1].value * 7
                for j in range(7):
                    physical_targets.extend([control + j, target + j])
            concatenated_circuit.append("CX", physical_targets, args)
            continue

        # Logical measurement and reset
        if name in ("M", "MR"):
            physical_targets = []
            groups = []
            for target in targets:
                base = target.value * 7
                groups.append(list(range(output_record_count, output_record_count + 7)))
                physical_targets.extend(base + j for j in range(7))
                output_record_count += 7
            concatenated_circuit.append("M", physical_targets, args)
            record_map.extend(groups)
            input_record_count += len(targets)

            if name == "MR":
                concatenated_circuit.append("TICK")
                bases = [t.value * 7 for t in targets]
                logical_qubits(concatenated_circuit, bases)
            continue

        # Preserve all other instructions
        concatenated_circuit.append(instruction)

    return concatenated_circuit

def concatenate_circuit(circuit, level):
    """Create the concatenated circuit through recursion."""
    # Level 0 = L1 Steane code [[7, 1, 3]]
    # Level 1 = L2 Steane code [[49, 1, 9]]
    # Level 2 = L3 Steane code [[343, 1, 27]]
    result = circuit.copy()
    for _ in range(level):
        result = concatenate_block(result)
    return result