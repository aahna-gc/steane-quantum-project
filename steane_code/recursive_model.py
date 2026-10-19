import stim

"""Steane code data from the level-1 circuit, basis for recursion"""
steane_checks = [(3, 4, 5, 6), (1, 2, 5, 6), (0, 2, 4, 6)]
gate_order = [0, None, 1, 2, None, 3]

def make_context(num_blocks, levels, ec_in_gadgets, top_ec_blocks):
    """Initializes the information required to construct the concatenated Steane code."""
    offset = num_blocks * 7 ** levels
    pool = {}
    for k in range(1, levels + 1):
        size = 7 ** (k-1)
        pool[k] = [offset + j * size for j in range(12)]
        offset += 12 * size
    return {
        "circuit": stim.Circuit(),
        "count": 0,
        "depth": 0,
        "levels": levels,
        "ec_in_gadgets": ec_in_gadgets,
        "top_ec": set(range(num_blocks)) if top_ec_blocks is None else set(top_ec_blocks),
        "pool": pool,
        "state": {}
    }

def block_state(context, level, base):
    """Retrieve a record of the error-correction state for one logical block."""
    key = (level, base)
    if key not in context["state"]:
        context["state"][key] = {name: [set() for _ in range(3)] for name in ("sz", "sx", "fz", "fx")}
    return context["state"][key]

def create_block(level, base, q):
    """Create a logical qubit block: each block has 7 data qubits."""
    return base + q * 7 ** (level-1)

def add_detector(context, records, name="DETECTOR", args=()):
    """Add a detector to the concatenated code."""
    context["circuit"].append(name, [stim.target_rec(i-context["count"]) for i in sorted(records)], list(args))

def logical_qubits(context, level, base):
    """Prepare the |0> state."""
    if level == 0:
        context["circuit"].append("R", [base])
        return
    for q in range(7):
        logical_qubits(context, level-1, create_block(level, base, q))
    context["circuit"].append("TICK")

    x_syndromes, x_flags = measure_checks(context, level, base, "x")
    for flag in x_flags:
        add_detector(context, flag)

    state = block_state(context, level, base)
    state["sz"] = [set() for _ in range(3)]
    state["fz"] = [set() for _ in range(3)]
    state["sx"] = list(x_syndromes)
    state["fx"] = list(x_syndromes)

def logical_h(context, level, base):
    """Create a transversal logical H gate."""
    if level == 0:
        context["circuit"].append("H", [base])
        return
    for q in range(7):
        logical_h(context, level - 1, create_block(level, base, q))
    state = block_state(context, level, base)
    state["sz"], state["sx"] = state["sx"], state["sz"]
    state ["fz"], state["fx"] = state["fx"], state["fz"]
    error_correction(context, level, base)

def logical_cx(context, level, control, target):
    """Create a transversal logical CNOT gate. X flows control -> target, whilst Z flows target -> control."""
    if level == 0:
        context["circuit"].append("CX", [control, target])
        return
    for q in range(7):
        logical_cx(context, level - 1, create_block(level, control, q), create_block(level, target, q))
    control_state = block_state(context, level, control)
    target_state = block_state(context, level, target)
    for i in range(3):
        for key in ("sz", "fz"):
            target_state[key][i] = target_state[key][i] ^ control_state[key][i]
        for key in ("sx", "fx"):
            control_state[key][i] = control_state[key][i] ^ target_state[key][i]
    error_correction(context, level, control)
    error_correction(context, level, target)

def logical_measure(context, level, base):
    """Z-basis logical measurement, returning a set of physical record indices whose XOR is the outcome."""
    if level == 0:
        context["circuit"].append("M", [base])
        context["count"] += 1
        return {context["count"] - 1}

    outcomes = [logical_measure(context, level - 1, create_block(level, base, q)) for q in range(7)]
    state = block_state(context, level, base)

    for i, check in enumerate(steane_checks):
        parity = set(state["sz"][i])
        for q in check:
            parity = parity ^ outcomes[q]
        add_detector(context, parity)

    result = set()
    for outcome in outcomes:
        result = result ^ outcome
    for i in range(3):
        result = result ^ state["fz"][i]
    return result

def measure_checks(context, level, base, kind):
    """Performs a flagged-stabilizer measurement on one block."""
    circuit = context["circuit"]
    context["depth"] += 1
    pool = context["pool"][level]
    if kind == "z":
        ancillas, flags = pool[0:3], pool[6:9]
        for a in ancillas:
            logical_qubits(context, level - 1, a)
        for f in flags:
            logical_qubits(context, level - 1, f)
            logical_h(context, level - 1, f)

    else:
        ancillas, flags = pool[3:6], pool[9:12]
        for a in ancillas:
            logical_qubits(context, level - 1, a)
            logical_h(context, level - 1, a)
        for f in flags:
            logical_qubits(context, level - 1, f)
    circuit.append("TICK")

    for s in gate_order:
        for i in range(3):
            if kind == "z":
                control = flags[i] if s is None else create_block(level, base, steane_checks[i][s])
                logical_cx(context, level - 1, control, ancillas[i])
            else:
                target = flags[i] if s is None else create_block(level, base, steane_checks[i][s])
                logical_cx(context, level - 1, ancillas[i], target)
        circuit.append("TICK")

    for block in (flags if kind == "z" else ancillas):
        logical_h(context, level - 1, block)
    circuit.append("TICK")
    syndromes = [logical_measure(context, level - 1, a) for a in ancillas]
    flag_outcomes = [logical_measure(context, level - 1, f) for f in flags]
    circuit.append("TICK")
    context["depth"] -= 1
    return syndromes, flag_outcomes

def error_correction(context, level, base):
    """Records flag outcomes by measuring both X and Z type stabilizers."""
    if level == 0:
        return
    if context["depth"] and not context["ec_in_gadgets"]:
        return
    if level == context["levels"] and (base // 7 ** context["levels"]) not in context["top_ec"]:
        return

    z_syndromes, z_flags = measure_checks(context, level, base, "z")
    x_syndromes, x_flags = measure_checks(context, level, base, "x")
    for flag in z_flags + x_flags:
        add_detector(context, flag)
    state = block_state(context, level, base)
    for i in range(3):
        add_detector(context, z_syndromes[i] ^ state["sz"][i])
        add_detector(context, x_syndromes[i] ^ state["sx"][i])
        state["sz"][i] = z_syndromes[i]
        state["sx"][i] = x_syndromes[i]

def concatenate_block(circuit, level, ec_in_gadgets=False, top_ec_blocks=None):
    """Concatenate a block."""
    context = make_context(circuit.num_qubits, level, ec_in_gadgets, top_ec_blocks)
    concatenated_circuit = context["circuit"]
    size = 7 ** level

    record_map = []
    input_record_count = 0

    for instruction in circuit:
        name = instruction.name
        targets = instruction.targets_copy()
        args = instruction.gate_args_copy()

        if name == "TICK":
            concatenated_circuit.append("TICK")
            continue

        # Expand detector and observables.
        if name in ("DETECTOR", "OBSERVABLE_INCLUDE"):
            expanded = set()
            for target in targets:
                input_index = input_record_count + target.value
                expanded = expanded ^ record_map[input_index]
            add_detector(context, expanded, name, args)
            continue

        # Prepare the |0_L> state (reset)
        if name == "R":
            for target in targets:
                logical_qubits(context, level, target.value * size)
            continue

        # Transversal logical H
        if name == "H":
            for target in targets:
                logical_h(context, level, target.value * size)
            continue

        # Transversal logical CNOT
        if name == "CX":
            for i in range(0, len(targets), 2):
                logical_cx(context, level, targets[i].value * size, targets[i + 1].value * size)
            continue

        # Logical measurement and reset
        if name in ("M", "MR"):
            blocks = [target.value * size for target in targets]
            for base in blocks:
                record_map.append(logical_measure(context, level, base))
                input_record_count += 1

            if name == "MR":
                concatenated_circuit.append("TICK")
                for base in blocks:
                    logical_qubits(context, level, base)
            continue

    return concatenated_circuit

def concatenate_circuit(circuit, level, ec_in_gadgets=False, top_ec_blocks=None):
    """Create the concatenated circuit."""
    # Level 0 = L1 Steane code [[7, 1, 3]]
    # Level 1 = L2 Steane code [[49, 1, 9]]
    # Level 2 = L3 Steane code [[343, 1, 27]]
    # The levels are built recursively inside the gadgets, so one call to concatenate_block does it all.
    if level == 0:
        return circuit.copy()
    return concatenate_block(circuit, level, ec_in_gadgets, top_ec_blocks)