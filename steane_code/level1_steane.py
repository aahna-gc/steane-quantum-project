import stim

"""Encode the stabilizers and measurements."""
def steane_stabilizer_round(circuit, first_round=False):
    # Z-stabilizers (bit-flip errors)
    z_stabilizers = [
        (3, 4, 5, 6),
        (1, 2, 5, 6),
        (0, 2, 4, 6)
    ]

    # Ancillas used for the z-stabilizers -> act as the target.
    z_ancillas = [7, 8, 9]

    # Entangle the ancilla and stabilizer together.
    for k in range(4):
        for ancilla, stabilizer in zip(z_ancillas, z_stabilizers):
            circuit.append("CX", [stabilizer[k], ancilla])
        circuit.append("TICK")
    circuit.append("MR", z_ancillas)
    circuit.append("TICK")

    # X-stabilizers (phase-flip errors)
    x_stabilizers = [
        (3, 4, 5, 6),
        (1, 2, 5, 6),
        (0, 2, 4, 6)
    ]

    # Ancillas used for the x-stabilizers -> act as the control.
    x_ancillas = [10, 11, 12]

    # Entangle the ancilla and stabilizer together.
    circuit.append("H", x_ancillas)
    circuit.append("TICK")
    for k in range(4):
        for ancilla, stabilizer in zip(x_ancillas, x_stabilizers):
            circuit.append("CX", [ancilla, stabilizer[k]])
        circuit.append("TICK")
    circuit.append("H", x_ancillas)
    circuit.append("TICK")
    circuit.append("MR", x_ancillas)
    circuit.append("TICK")

    if first_round:
        for i in range(6):
            circuit.append("DETECTOR", [stim.target_rec(-1-i)])
    else:
        # Get the X-stabilizer measurements.
        for i in range(3):
            circuit.append("DETECTOR", [stim.target_rec(-1-i), stim.target_rec(-7-i)])
        # Get the Z-stabilizer measurements.
        for i in range(3):
            circuit.append("DETECTOR", [stim.target_rec(-4-i), stim.target_rec(-10-i)])
    
    return circuit

"""Start with the [[7, 1, 3]] Steane Code to use as a base for concatenation later."""
def level_1_steane(rounds):
    circuit = stim.Circuit()

    # Initialize all data qubits (0-6) and the ancilla qubits (7-12)
    circuit.append("R", range(13))
    circuit.append("TICK")

    # Encode the Steane code.
    circuit.append("H", [0, 1, 3])
    circuit.append("TICK")
    for q in [0, 1]:
        circuit.append("CX", [q, 2])
    circuit.append("TICK")
    for q in [0, 3]:
        circuit.append("CX", [q, 4])
    circuit.append("TICK")
    for q in [1, 3]:
        circuit.append("CX", [q, 5])
    circuit.append("TICK")
    for q in [0, 1, 3]:
        circuit.append("CX", [q, 6])
    circuit.append("TICK")

    # Run QEC rounds.
    for runs in range(rounds):
        steane_stabilizer_round(circuit, first_round=(runs==0))

    # Returns Z logical.
    circuit.append("M", range(7))
    circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1-i) for i in range(7)], 0)
    return circuit