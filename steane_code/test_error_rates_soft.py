# also aahnas file, ignore this 

import csv
import time
from pathlib import Path

from level1_steane import level_1_steane
from soft_noise_helpers import add_circuit_noise


# Experiment settings
SHOTS = 10_000
WARMUP_SHOTS = 100
ERROR_RATES = [1e-4, 3e-4, 1e-3, 3e-3]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_CSV = PROJECT_ROOT / "steane_code" / "level1_error_rate_results.csv"


def run_error_rate_test():
    print("Level-1 Steane circuit: [[7,1,3]]")
    print(f"Measured shots per error rate: {SHOTS}")
    print(f"Warm-up shots per error rate: {WARMUP_SHOTS}")
    print("No decoder is applied in this experiment.\n")

    # Build the noiseless circuit once.
    start = time.perf_counter()
    noiseless_circuit = level_1_steane(1)
    circuit_build_time = time.perf_counter() - start

    print(f"Circuit construction time: {circuit_build_time:.6f} s")
    print(f"Number of qubits: {noiseless_circuit.num_qubits}\n")

    results = []

    print(
        f"{'Physical rate':<16}"
        f"{'Raw flip rate':<18}"
        f"{'Sampling (s)':<16}"
        f"{'Setup (s)':<12}"
    )
    print("-" * 62)

    for p in ERROR_RATES:
        # Construct the noisy circuit for this physical error probability.
        noisy_circuit = add_circuit_noise(
            noiseless_circuit,
            p_idle=p,
            p_1gate=p,
            p_2gate=p,
            p_measure=p,
            p_reset=p,
        )

        # Time sampler compilation separately.
        start = time.perf_counter()
        sampler = noisy_circuit.compile_detector_sampler()
        compile_time = time.perf_counter() - start

        # Warm up the sampler. This sample is not included in the results.
        sampler.sample(
            shots=WARMUP_SHOTS,
            separate_observables=True,
        )

        # Time the measured sampling call only.
        start = time.perf_counter()
        detection_events, observable_flips = sampler.sample(
            shots=SHOTS,
            separate_observables=True,
        )
        sampling_time = time.perf_counter() - start

        raw_flip_rate = float(observable_flips.mean())

        results.append({
            "physical_error_rate": p,
            "shots": SHOTS,
            "warmup_shots": WARMUP_SHOTS,
            "num_qubits": noiseless_circuit.num_qubits,
            "num_detectors": detection_events.shape[1],
            "raw_observable_flip_rate": raw_flip_rate,
            "circuit_construction_seconds": circuit_build_time,
            "sampler_compilation_seconds": compile_time,
            "sampling_seconds": sampling_time,
            "decoder_runtime_seconds": "",
        })

        print(
            f"{p:<16.1e}"
            f"{raw_flip_rate:<18.6f}"
            f"{sampling_time:<16.6f}"
            f"{compile_time:<12.6f}"
        )

    # Save results to CSV for later analysis.
    with OUTPUT_CSV.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=results[0].keys(),
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {OUTPUT_CSV}")
    print("Raw observable-flip rates are not decoded logical failure rates.")
    print("Decoder runtime is blank because no decoder is called here.")


if __name__ == "__main__":
    run_error_rate_test()
