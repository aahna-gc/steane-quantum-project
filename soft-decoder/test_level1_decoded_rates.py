# python soft-decoder/soft-decoder/test_level1_decoded_rates.py

import csv
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STEANE_DIR = PROJECT_ROOT / "steane_code"

sys.path.insert(0, str(STEANE_DIR))

from level1_steane import level_1_steane
from soft_noise_helpers import add_circuit_noise
from soft_decoder import decode_steane_syndrome

SHOTS = 10_000
WARMUP_SHOTS = 100
ERROR_RATES = [1e-4, 3e-4, 1e-3, 3e-3]

OUTPUT_CSV = Path(__file__).resolve().parent / "level1_decoded_results.csv"


def decode_one_shot(detectors, observable_flip):
    # Detector order in Chloe's first round:
    # [X2, X1, X0, Z2, Z1, Z0]
    #
    # The decoder expects stabilizer order:
    # [stabilizer 0, stabilizer 1, stabilizer 2]

    x_stabilizer_syndrome = tuple(
        int(v) for v in detectors[2::-1]
    )
    z_stabilizer_syndrome = tuple(
        int(v) for v in detectors[5:2:-1]
    )

    # Z-stabilizer syndrome detects X errors.
    # X-stabilizer syndrome detects Z errors.
    result = decode_steane_syndrome(
        x_syndrome=z_stabilizer_syndrome,
        z_syndrome=x_stabilizer_syndrome,
    )

    if result["status"] != "decoded":
        return int(observable_flip), False

    correction = result["correction"]

    # A correction containing an odd number of X/Y operators
    # flips the final logical-Z measurement outcome.
    correction_flips_logical_z = sum(
        pauli in ("X", "Y") for pauli in correction
    ) % 2

    corrected_observable = (
        int(observable_flip) ^ correction_flips_logical_z
    )

    return corrected_observable, True


def run():
    print("Level-1 Steane syndrome-decoding benchmark")
    print(f"Shots per physical error rate: {SHOTS}")
    print("Building the circuit...")

    noiseless_circuit = level_1_steane(1)

    print(f"Qubits: {noiseless_circuit.num_qubits}")
    print(f"Detectors: {noiseless_circuit.num_detectors}")
    print(f"Observables: {noiseless_circuit.num_observables}\n")

    results = []

    for p in ERROR_RATES:
        print(f"Testing physical error rate p = {p:.1e}")

        noisy_circuit = add_circuit_noise(
            noiseless_circuit,
            p_idle=p,
            p_1gate=p,
            p_2gate=p,
            p_measure=p,
            p_reset=p,
        )

        sampler = noisy_circuit.compile_detector_sampler()

        sampler.sample(
            shots=WARMUP_SHOTS,
            separate_observables=True,
        )

        detection_events, observable_flips = sampler.sample(
            shots=SHOTS,
            separate_observables=True,
        )

        failures = 0
        unresolved = 0

        start = time.perf_counter()

        for i in range(SHOTS):
            corrected, decoded = decode_one_shot(
                detection_events[i],
                observable_flips[i, 0],
            )

            if not decoded:
                unresolved += 1

            if corrected:
                failures += 1

        decoder_runtime = time.perf_counter() - start

        failure_rate = failures / SHOTS

        result = {
            "physical_error_rate": p,
            "shots": SHOTS,
            "decoded_logical_failures": failures,
            "decoded_logical_failure_rate": failure_rate,
            "decoder_runtime_seconds": decoder_runtime,
            "decoder_microseconds_per_shot": (
                decoder_runtime / SHOTS * 1_000_000
            ),
            "unresolved_shots": unresolved,
        }

        results.append(result)

        print(f"  Decoded logical failure rate: {failure_rate:.6f}")
        print(f"  Logical failures: {failures}/{SHOTS}")
        print(f"  Decoder runtime: {decoder_runtime:.6f} s")
        print(
            "  Average decoder time: "
            f"{decoder_runtime / SHOTS * 1_000_000:.3f} us/shot"
        )
        print(f"  Unresolved shots: {unresolved}\n")

    with OUTPUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=results[0].keys(),
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to: {OUTPUT_CSV}")
    print(
        "Important: this uses a single-round, single-qubit "
        "syndrome decoder, not the full soft-information decoder."
    )


if __name__ == "__main__":
    run()
