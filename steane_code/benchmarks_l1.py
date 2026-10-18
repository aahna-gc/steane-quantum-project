from level1_steane import level_1_steane
from noise import add_circuit_noise

"""Detector Measurements"""
level1_circuit = level_1_steane(1)
noise_levels = [1e-3, 3e-4, 1e-4, 3e-5, 1e-5]
runs = 100_000
for p in noise_levels:
    noisy_circuit = add_circuit_noise(
        level_1_steane(rounds=1),
        p_idle=p,
        p_1gate=p,
        p_2gate=p,
        p_measure=p,
        p_reset=p,
    )
    sampler = noisy_circuit.compile_detector_sampler()
    detection_events, observable_flips = sampler.sample(
    shots=runs,
    separate_observables=True
    )

    print("Detection events:", detection_events.shape)
    print("Observable flips:", observable_flips.shape)
    print("Logical error rate:", observable_flips.mean())
    print("Total detector events:", detection_events.sum())
    print("Total logical flips:", observable_flips.sum())
    print(f"p={p:.1e}, logical error rate={observable_flips.mean():.6g}")
    print(len(noisy_circuit.shortest_graphlike_error()))