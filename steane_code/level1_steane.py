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
    # Z-flag qubits to encode for fault-tolerance -> |+> control and measured in the X-basis
    z_flags = [13, 14,15]

    # Gate order for each check to ensure errors haven't propagated
    gate_order = [0, None, 1, 2, None, 3]

    # Entangle the ancilla and stabilizer together whilst accounting for the flag qubits
    circuit.append("H", z_flags)
    circuit.append("TICK")
    for s in gate_order:
        for ancilla, flag, stabilizer in zip(z_ancillas, z_flags, z_stabilizers):
            if s is None:
                circuit.append("CX", [flag, ancilla])
            else:
                circuit.append("CX", [stabilizer[s], ancilla])
        circuit.append("TICK")
    circuit.append("H", z_flags)
    circuit.append("TICK")
    circuit.append("MR", z_ancillas)
    circuit.append("MR", z_flags)
    circuit.append("TICK")

    # X-stabilizers (phase-flip errors)
    x_stabilizers = [
        (3, 4, 5, 6),
        (1, 2, 5, 6),
        (0, 2, 4, 6)
    ]

    # Ancillas used for the x-stabilizers -> act as the control.
    x_ancillas = [10, 11, 12]
    # X-flag qubits to encode for fault tolerance -> |0> target and measured in the Z-basis
    x_flags = [16, 17, 18]

    # Entangle the ancilla and stabilizer together whilst accounting for fault tolerance with the flag qubits.
    circuit.append("H", x_ancillas)
    circuit.append("TICK")
    for s in gate_order:
        for ancilla, flag, stabilizer in zip(x_ancillas, x_flags, x_stabilizers):
            if s is None:
                circuit.append("CX", [ancilla, flag])
            else:
                circuit.append("CX", [ancilla, stabilizer[s]])
    circuit.append("H", x_ancillas)
    circuit.append("TICK")
    circuit.append("MR", x_ancillas)
    circuit.append("MR", x_flags)
    circuit.append("TICK")

    # Get the flag measurements to ensure fault tolerance -> flags should always read 0.
    for i in range(3):
        # Z-flags
        circuit.append("DETECTOR", [stim.target_rec(-9 + i)])
        # X-flags
        circuit.append("DETECTOR", [stim.target_rec(-3 + i)])

    if first_round:
        # Get Z-stabilizer measurements; X-stabilizer measurements are random so do not have to be measured.
        for i in range(3):
            circuit.append("DETECTOR", [stim.target_rec(-12 + i)])
    else:
        # Get the X-stabilizer measurements and compare to previous round.
        for i in range(3):
            circuit.append("DETECTOR", [stim.target_rec(-6 + i), stim.target_rec(-18 + i)])
        # Get the Z-stabilizer measurements and compare to previous round.
        for i in range(3):
            circuit.append("DETECTOR", [stim.target_rec(-12 + i), stim.target_rec(-24 + i)])
    
    return circuit

"""Start with the [[7, 1, 3]] Steane Code to use as a base for concatenation later."""
def level_1_steane(rounds):
    circuit = stim.Circuit()

    # Initialize all data qubits (0-6), the ancilla qubits (7-12), and flag qubits (13-18)
    circuit.append("R", range(19))
    circuit.append("TICK")

    # Run QEC rounds.
    for runs in range(rounds):
        steane_stabilizer_round(circuit, first_round=(runs==0))

    # Returns Z logical.
    circuit.append("M", range(7))
    # Final Z-stabilizer check coparing data parities and last round's Z syndromes.
    z_stabilizers = [(3, 4, 5, 6), (1, 2, 5, 6), (0, 2, 4, 6)]
    for j, stab in enumerate(z_stabilizers):
        circuit.append("DETECTOR", [stim.target_rec(-7 + q) for q in stab] + [stim.target_rec(-19 + j)])
    circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1-i) for i in range(7)], 0)
    return circuit