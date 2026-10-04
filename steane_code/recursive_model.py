import stim

"""Start with the [[7, 1, 3]] Steane Code to use as a base for concatenation later."""
def level_1_steane():
    circuit = stim.Circuit()

    # Initialize all data qubits (0-6) and the ancilla qubits (7-12)
    circuit.append("R", range(13))

    # Encode the Steane code.
    circuit.append("H", [0, 1, 3])
    for q in [0, 1]:
        circuit.append("CNOT", [q, 2])
    for q in [0, 3]:
        circuit.append("CNOT", [q, 4])
    for q in [1, 3]:
        circuit.append("CNOT", [q, 5])
    for q in [0, 1, 3]:
        circuit.append("CNOT", [q, 6])

    return circuit

"""Encode the stabilizers and measurements."""
def steane_stabilizer_round(circuit):
    # Z-stabilizers (bit-flip errors)
    z_stabilizers = [
        (3, 4, 5, 6),
        (1, 2, 5, 6),
        (0, 2, 4, 6)
    ]

    # Ancillas used for the z-stabilizers -> act as the target.
    z_ancillas = [7, 8, 9]

    # Entangle the ancilla and stabilizer together.
    for ancilla, stabilizer in zip(z_ancillas, z_stabilizers):
        for data in stabilizer:
            circuit.append("CX", [data, ancilla])
    circuit.append("MR", z_ancillas)

    # X-stabilizers (phase-flip errors)
    x_stabilizers = [
        (3, 4, 5, 6),
        (1, 2, 5, 6),
        (0, 2, 4, 6)
    ]

    # Ancillas used for the x-stabilizers -> act as the control.
    x_ancillas = [10, 11, 12]

    # Entangle the ancilla and stabilizer together.
    for ancilla, stabilizer in zip(x_ancillas, x_stabilizers):
        circuit.append("H", [ancilla])
        for data in stabilizer:
            circuit.append("CX", [ancilla, data])
        circuit.append("H", [ancilla])
    circuit.append("MR", x_ancillas)

    return circuit