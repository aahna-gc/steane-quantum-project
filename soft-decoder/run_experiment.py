"""
Run the L=1 soft-decoder experiment and write results to CSV.

Put this file next to message_passing_decoder.py (in soft-decoder/) and run from the repo root:

    python soft-decoder/run_experiment.py --quick     # fast smoke test, not for the paper
    python soft-decoder/run_experiment.py             # full run
    python soft-decoder/run_experiment.py --force     # run even if circuit distance < 3

What it does
------------
1. Builds the noisy circuit for each physical error rate p (partner's circuit + noise code).
2. Checks the circuit's graphlike distance and refuses to run if it is < 3 (override: --force),
   because below 3 the logical failure rate is set by the circuit, not the decoder.
3. Decodes every single fault in the detector error model and reports how many BP gets wrong.
4. For each p, samples in batches until TARGET_FAILURES logical failures (or MAX_SHOTS),
   so every point has a meaningful confidence interval.  Sampling and decoding are timed
   separately.  Results go to results_L1_soft.csv.

To run a different decoder on the same circuit/noise/shots, give it a
`decode_batch(detection_events) -> (num_shots, num_observables) bool` method and swap it in
`make_decoder`.
"""
from __future__ import annotations

import csv
import math
import sys
import time
from pathlib import Path

import numpy as np

# ------------------------------- settings ------------------------------------------
ROUNDS = 3                      # syndrome rounds; use the same value for every decoder
P_LIST = [0.0005, 0.001, 0.002, 0.004, 0.008]
MAX_ITER = 50
TARGET_FAILURES = 200           # stop a point once this many logical failures are seen
MAX_SHOTS = 5_000_000
BATCH = 200_000
SEED = 1234
ADD_FINAL_Z_DETECTORS = False    # set False once the circuit already has final-round detectors
CSV_PATH = "results_L1_soft.csv"
# ------------------------------------------------------------------------------------


def wilson_interval(k: int, n: int, z: float = 1.96):
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 1.0
    phat = k / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def single_fault_test(decoder):
    """Decode every single fault in the DEM; report how many BP gets wrong."""
    S = decoder.H.toarray().T.astype(int)       # (faults, detectors)
    obs = decoder.L.toarray().T.astype(bool)    # (faults, observables)
    has_syndrome = S.any(axis=1)
    out = {
        "faults": int(len(S)),
        "undetectable_faults": int((~has_syndrome).sum()),
        "undetectable_logical_faults": int((~has_syndrome & obs.any(axis=1)).sum()),
    }
    if has_syndrome.any():
        pred, info = decoder.decode_batch(S[has_syndrome], return_details=True)
        wrong = np.any(pred != obs[has_syndrome], axis=1)
        out["detectable_faults"] = int(has_syndrome.sum())
        out["bp_nonconverged"] = int((~info["converged"]).sum())
        out["bp_wrong"] = int(wrong.sum())
    return out


def run_point(build_circuit, make_decoder, p: float, point_index: int,
              target_failures: int, max_shots: int, batch: int):
    circuit = build_circuit(p)
    t0 = time.perf_counter()
    decoder = make_decoder(circuit)
    build_time = time.perf_counter() - t0
    sampler = circuit.compile_detector_sampler(seed=SEED + point_index)

    shots = failures = nonconv = fail_nonconv = 0
    iters_sum = 0
    t_sample = t_decode = 0.0
    while failures < target_failures and shots < max_shots:
        n = min(batch, max_shots - shots)

        t0 = time.perf_counter()
        det, obs = sampler.sample(n, separate_observables=True)
        t_sample += time.perf_counter() - t0

        t0 = time.perf_counter()
        preds, info = decoder.decode_batch(det, return_details=True)
        t_decode += time.perf_counter() - t0

        failed = np.any(preds != obs, axis=1)
        conv = info["converged"]
        shots += n
        failures += int(failed.sum())
        nonconv += int((~conv).sum())
        fail_nonconv += int(failed[~conv].sum())
        iters_sum += int(info["iterations"].sum())

    lo, hi = wilson_interval(failures, shots)
    return {
        "p": p,
        "shots": shots,
        "failures": failures,
        "P_L": failures / shots,
        "P_L_ci95_low": lo,
        "P_L_ci95_high": hi,
        "bp_nonconverged_fraction": nonconv / shots,
        "failures_in_nonconverged": fail_nonconv,
        "mean_bp_iterations": iters_sum / shots,
        "decode_time_s": t_decode,
        "decode_time_per_shot_us": 1e6 * t_decode / shots,
        "sample_time_s": t_sample,
        "build_time_s": build_time,
        "graph_edges": getattr(decoder, "n_edges", ""),
        "max_iter": MAX_ITER,
        "rounds": ROUNDS,
        "reached_target": failures >= target_failures,
    }


def main():
    import contextlib
    import io

    import stim  # noqa: F401  (fails early with a clear error if stim is missing)

    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))                        # message_passing_decoder
    sys.path.insert(0, str(here.parent / "steane_code"))  # partner's code

    from message_passing_decoder import SumProductBPDecoder, with_final_z_detectors
    from level1_steane import level_1_steane

    # noise_and_benchmarks.py runs its own script at import time; silence it.
    with contextlib.redirect_stdout(io.StringIO()):
        from noise_and_benchmarks import add_circuit_noise

    quick = "--quick" in sys.argv
    force = "--force" in sys.argv
    target = 20 if quick else TARGET_FAILURES
    max_shots = 200_000 if quick else MAX_SHOTS
    batch = 50_000 if quick else BATCH

    def build_circuit(p):
        c = level_1_steane(ROUNDS)
        if ADD_FINAL_Z_DETECTORS:
            c = with_final_z_detectors(c)
        return add_circuit_noise(c, p_idle=p, p_1gate=p, p_2gate=p, p_measure=p, p_reset=p)

    def make_decoder(circuit):
        return SumProductBPDecoder.from_circuit(circuit, max_iter=MAX_ITER)

    # ---- diagnostics at a reference p ----
    ref = build_circuit(1e-3)
    try:
        dist = len(ref.shortest_graphlike_error())
    except Exception as exc:  # noqa: BLE001
        dist = None
        print("distance check failed:", exc)
    print(f"graphlike circuit distance (upper bound on true distance): {dist}")
    if dist is not None and dist < 3 and not force:
        print(
            "\nStopping: distance < 3, so P_L is limited by the circuit (some single faults flip "
            "the logical with no detection event), not by the decoder.\n"
            "Fix the circuit, or rerun with --force to produce pipeline-test numbers only."
        )
        sys.exit(1)

    print("single-fault test:", single_fault_test(make_decoder(ref)))
    print(f"rounds={ROUNDS}  max_iter={MAX_ITER}  target_failures={target}  quick={quick}\n")

    rows = []
    for i, p in enumerate(P_LIST):
        row = run_point(build_circuit, make_decoder, p, i, target, max_shots, batch)
        rows.append(row)
        print(
            f"p={p:<7} P_L={row['P_L']:.3e} [{row['P_L_ci95_low']:.2e}, {row['P_L_ci95_high']:.2e}]  "
            f"shots={row['shots']}  nonconv={row['bp_nonconverged_fraction']:.4f}  "
            f"iters={row['mean_bp_iterations']:.2f}  "
            f"decode={row['decode_time_per_shot_us']:.1f} us/shot"
            + ("" if row["reached_target"] else "  (hit MAX_SHOTS before target)")
        )

    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {CSV_PATH}")


if __name__ == "__main__":
    main()
