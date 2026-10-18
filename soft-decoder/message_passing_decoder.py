"""
Soft-information message passing decoder (sum-product belief propagation, LLR form)
for circuit-level noise, built directly from a Stim detector error model (DEM).

Graph
-----
    variable nodes : fault mechanisms        (one per `error(p) ...` line of the DEM)
    check nodes    : detectors               (parity constraints on detection events)
    edge (c, v)    : fault v flips detector c
    priors         : p_v read from the DEM

Convention
----------
    LLR = ln( P(bit = 0) / P(bit = 1) ).  Positive LLR -> "fault did not happen".
    Prior LLR of fault v:  ln((1 - p_v) / p_v).

The decoder estimates which faults fired, then maps that estimate to a prediction of the
logical observable flip(s) through the observable matrix.

Cost accounting
---------------
Decoding is vectorised over shots, but every shot stops as soon as its own estimate
reproduces its syndrome: converged shots are removed from the active batch after each
iteration, so work (and runtime) scales with how hard each shot is, not with the hardest
shot in a chunk.  Per-shot iteration counts are returned as a hardware-independent cost
measure alongside wall-clock decode time.
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
import stim

_EPS = 1e-300           # floor for |tanh| before taking logs
_TANH_CLIP = 1 - 1e-12  # keeps arctanh finite


# ----------------------------------------------------------------------------------
# DEM -> matrices
# ----------------------------------------------------------------------------------
def dem_to_matrices(dem: stim.DetectorErrorModel):
    """
    Convert a DEM into (H, L, priors).

        H      : (num_detectors   x num_faults) binary sparse check matrix
        L      : (num_observables x num_faults) binary sparse observable matrix
        priors : (num_faults,) fault probabilities

    Use `decompose_errors=False` when building the DEM so each `error` line is one
    full fault mechanism (decomposed `^` pieces are merged here if present).
    """
    dem = dem.flattened()
    det_r, det_c, obs_r, obs_c, probs = [], [], [], [], []
    n_faults = 0
    for inst in dem:
        if inst.type != "error":
            continue
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                det_r.append(t.val)
                det_c.append(n_faults)
            elif t.is_logical_observable_id():
                obs_r.append(t.val)
                obs_c.append(n_faults)
            # separators ("^") are ignored: pieces are merged into one mechanism
        probs.append(inst.args_copy()[0])
        n_faults += 1

    def _binary(rows, cols, n_rows):
        m = sp.coo_matrix(
            (np.ones(len(rows), dtype=np.int64), (rows, cols)),
            shape=(n_rows, n_faults),
        ).tocsr()
        m.data %= 2            # a detector hit twice by one fault cancels
        m.eliminate_zeros()
        return m

    H = _binary(det_r, det_c, dem.num_detectors)
    L = _binary(obs_r, obs_c, dem.num_observables)
    return H, L, np.asarray(probs, dtype=float)


# ----------------------------------------------------------------------------------
# Decoder
# ----------------------------------------------------------------------------------
class SumProductBPDecoder:
    """Sum-product BP in LLR form on the detector-fault graph."""

    def __init__(self, H, L, priors, max_iter: int = 50, batch_size: int = 2048):
        self.H = sp.csr_matrix(H, dtype=np.int64)
        self.L = sp.csr_matrix(L, dtype=np.int64)
        priors = np.clip(np.asarray(priors, dtype=float), 1e-12, 1 - 1e-12)
        self.prior_llr = np.log((1 - priors) / priors)
        self.max_iter = max_iter
        self.batch_size = batch_size

        self.n_checks, self.n_vars = self.H.shape

        # Edge list of the Tanner graph.
        coo = self.H.tocoo()
        self.chk_idx = coo.row.astype(np.int64)   # check endpoint of each edge
        self.var_idx = coo.col.astype(np.int64)   # variable endpoint of each edge
        self.n_edges = len(self.chk_idx)
        eids = np.arange(self.n_edges)
        ones = np.ones(self.n_edges)
        # Sparse "sum over edges incident to a node" operators.
        self._sum_chk = sp.csr_matrix((ones, (self.chk_idx, eids)), shape=(self.n_checks, self.n_edges))
        self._sum_var = sp.csr_matrix((ones, (self.var_idx, eids)), shape=(self.n_vars, self.n_edges))

    @classmethod
    def from_dem(cls, dem: stim.DetectorErrorModel, **kwargs):
        H, L, priors = dem_to_matrices(dem)
        return cls(H, L, priors, **kwargs)

    @classmethod
    def from_circuit(cls, circuit: stim.Circuit, **kwargs):
        dem = circuit.detector_error_model(decompose_errors=False)
        return cls.from_dem(dem, **kwargs)

    # ------------------------------------------------------------------------------
    def _syndrome_matches(self, est: np.ndarray, synd: np.ndarray) -> np.ndarray:
        got = np.asarray(self.H @ est.astype(np.int64)) % 2
        return np.all(got == synd, axis=0)

    def _decode_chunk(self, syndromes: np.ndarray):
        """
        syndromes : (num_detectors, S) array of 0/1.
        Returns (fault_estimate (V,S) bool, posterior_llr (V,S), converged (S,) bool,
                 iterations (S,) int).

        iterations[s] = BP iterations shot s used: 0 if the priors alone already satisfy the
        syndrome, the iteration at which it converged, or max_iter if it never did.
        """
        V, E = self.n_vars, self.n_edges
        S = syndromes.shape[1]
        prior = self.prior_llr[:, None]

        # Iteration 0: hard decision from the priors alone.
        post0 = np.broadcast_to(prior, (V, S))
        est0 = post0 < 0
        done = self._syndrome_matches(est0, syndromes)
        final_est = est0.copy()
        final_post = post0.copy()
        iterations = np.zeros(S, dtype=np.int64)

        active = np.flatnonzero(~done)          # global indices of shots still iterating
        if active.size and self.max_iter > 0:
            synd = syndromes[:, active]
            synd_sign = 1.0 - 2.0 * synd
            m_vc = np.broadcast_to(prior[self.var_idx], (E, active.size)).copy()
            est = est0[:, active]
            post = post0[:, active].copy()

            for it in range(1, self.max_iter + 1):
                # ---- check -> variable (tanh rule, computed in the log domain) ----
                t = np.tanh(0.5 * m_vc)                                # (E,A)
                neg = (t < 0).astype(float)
                sgn = 1.0 - 2.0 * neg
                logabs = np.log(np.maximum(np.abs(t), _EPS))
                chk_logsum = self._sum_chk @ logabs                    # (C,A)
                chk_negcnt = self._sum_chk @ neg                       # (C,A)
                chk_sign = synd_sign * (1.0 - 2.0 * np.mod(chk_negcnt, 2))
                # Exclude each edge's own contribution ("extrinsic" message).
                val = chk_sign[self.chk_idx] * sgn * np.exp(chk_logsum[self.chk_idx] - logabs)
                val = np.clip(val, -_TANH_CLIP, _TANH_CLIP)
                m_cv = 2.0 * np.arctanh(val)                           # (E,A)

                # ---- variable -> check ----
                total = self._sum_var @ m_cv                           # (V,A)
                post = prior + total
                m_vc = post[self.var_idx] - m_cv

                # ---- hard decision + convergence test ----
                est = post < 0
                ok = self._syndrome_matches(est, synd)
                if ok.any():
                    idx = active[ok]
                    final_est[:, idx] = est[:, ok]
                    final_post[:, idx] = post[:, ok]
                    done[idx] = True
                    iterations[idx] = it

                    # Drop converged shots so later iterations only pay for the rest.
                    keep = ~ok
                    active = active[keep]
                    if active.size == 0:
                        break
                    synd = synd[:, keep]
                    synd_sign = synd_sign[:, keep]
                    m_vc = m_vc[:, keep]
                    est = est[:, keep]
                    post = post[:, keep]

            # Shots that never converged keep their last iterate.
            if active.size:
                final_est[:, active] = est
                final_post[:, active] = post
                iterations[active] = self.max_iter

        return final_est, final_post, done, iterations

    # ------------------------------------------------------------------------------
    def decode_batch(self, detection_events: np.ndarray, return_details: bool = False,
                     return_posteriors: bool = False):
        """
        detection_events : (num_shots, num_detectors) bool/0-1 array (as from Stim's sampler).

        Returns predicted observable flips, shape (num_shots, num_observables), bool.
        If return_details: returns (preds, info) where info is a dict with
            "converged"  : (num_shots,) bool
            "iterations" : (num_shots,) int, BP iterations used per shot
            "posterior_llr" : (num_shots, num_faults)   [only if return_posteriors=True]
        Posteriors are large (num_shots x num_faults), so they are opt-in.
        """
        det = np.asarray(detection_events).astype(np.int64)
        n_shots = det.shape[0]
        preds = np.zeros((n_shots, self.L.shape[0]), dtype=bool)
        conv = np.zeros(n_shots, dtype=bool)
        iters = np.zeros(n_shots, dtype=np.int64)
        posts = np.zeros((n_shots, self.n_vars)) if return_posteriors else None

        for start in range(0, n_shots, self.batch_size):
            stop = min(start + self.batch_size, n_shots)
            est, post, done, it = self._decode_chunk(det[start:stop].T)
            preds[start:stop] = (np.asarray(self.L @ est.astype(np.int64)) % 2).T.astype(bool)
            conv[start:stop] = done
            iters[start:stop] = it
            if return_posteriors:
                posts[start:stop] = post.T

        if return_details or return_posteriors:
            info = {"converged": conv, "iterations": iters}
            if return_posteriors:
                info["posterior_llr"] = posts
            return preds, info
        return preds

    def decode(self, detection_events: np.ndarray) -> np.ndarray:
        """Decode a single shot; returns the predicted observable flips, shape (num_observables,)."""
        return self.decode_batch(np.asarray(detection_events).reshape(1, -1))[0]


# ----------------------------------------------------------------------------------
# Experiment helpers
# ----------------------------------------------------------------------------------
def with_final_z_detectors(circuit: stim.Circuit) -> stim.Circuit:
    """
    OPTIONAL helper for circuits shaped like level_1_steane (final `M` of the 7 data qubits
    immediately followed by OBSERVABLE_INCLUDE, preceded by a round measuring 3 Z ancillas
    then 3 X ancillas).  Inserts the three final Z-stabilizer detectors that compare the
    data readout to the last round's Z ancillas.  Do NOT use on a circuit that already has them.
    """
    z_stabs = [(3, 4, 5, 6), (1, 2, 5, 6), (0, 2, 4, 6)]
    out = stim.Circuit()
    for inst in circuit:
        if inst.name == "OBSERVABLE_INCLUDE":
            for i, stab in enumerate(z_stabs):
                out.append("DETECTOR", [stim.target_rec(-13 + i)] + [stim.target_rec(-7 + q) for q in stab])
        out.append(inst)
    return out


def logical_failure_rate(circuit: stim.Circuit, num_shots: int, max_iter: int = 50, seed=None):
    """
    Sample `num_shots`, decode, and report logical failure rate plus cost measures.
    Sampling and decoding are timed separately; `decode_time_s` covers decode_batch only
    (graph construction is reported as `build_time_s`).
    """
    t0 = time.perf_counter()
    decoder = SumProductBPDecoder.from_circuit(circuit, max_iter=max_iter)
    build_time = time.perf_counter() - t0

    sampler = circuit.compile_detector_sampler(seed=seed)
    t0 = time.perf_counter()
    det, obs = sampler.sample(num_shots, separate_observables=True)
    sample_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    preds, info = decoder.decode_batch(det, return_details=True)
    decode_time = time.perf_counter() - t0

    failed = np.any(preds != obs, axis=1)
    conv = info["converged"]
    iters = info["iterations"]
    return {
        "shots": num_shots,
        "failures": int(failed.sum()),
        "logical_failure_rate": float(failed.mean()),
        "bp_converged_fraction": float(conv.mean()),
        "nonconverged_shots": int((~conv).sum()),
        "failures_converged": int(failed[conv].sum()),
        "failures_nonconverged": int(failed[~conv].sum()),
        "mean_bp_iterations": float(iters.mean()),
        "graph_edges": decoder.n_edges,
        "build_time_s": build_time,
        "sample_time_s": sample_time,
        "decode_time_s": decode_time,
        "decode_time_per_shot_s": decode_time / num_shots,
    }


# ----------------------------------------------------------------------------------
# Demo: uses the research partner's circuit and noise code as-is (nothing is duplicated)
# ----------------------------------------------------------------------------------
if __name__ == "__main__":
    import contextlib
    import io
    import sys
    from pathlib import Path

    # Partner's code lives in ../steane_code relative to this file.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "steane_code"))

    from level1_steane import level_1_steane

    # noise_and_benchmarks.py runs its own 100k-shot script at import time (it has no
    # `if __name__ == "__main__":` guard), so silence its prints while importing.
    with contextlib.redirect_stdout(io.StringIO()):
        from noise_and_benchmarks import add_circuit_noise

    ROUNDS = 1                       # same as the partner's benchmark script
    SHOTS = 100_000
    ADD_FINAL_Z_DETECTORS = False    # True only for the original circuit without final detectors

    def build(p):
        circuit = level_1_steane(ROUNDS)
        if ADD_FINAL_Z_DETECTORS:
            circuit = with_final_z_detectors(circuit)
        return add_circuit_noise(
            circuit, p_idle=p, p_1gate=p, p_2gate=p, p_measure=p, p_reset=p,
        )

    # Circuit-distance diagnostic (upper bound; graphlike search only).
    try:
        print("shortest graphlike logical error:", len(build(1e-3).shortest_graphlike_error()))
    except Exception as exc:  # noqa: BLE001
        print("distance check failed:", exc)

    for p in [0.0005, 0.001, 0.002, 0.005, 0.01]:
        r = logical_failure_rate(build(p), num_shots=SHOTS, max_iter=50, seed=0)
        print(
            f"p={p:<7} P_L={r['logical_failure_rate']:.5f}  "
            f"conv={r['bp_converged_fraction']:.3f}  "
            f"fail(conv/nonconv)={r['failures_converged']}/{r['failures_nonconverged']}  "
            f"iters={r['mean_bp_iterations']:.2f}  "
            f"decode={r['decode_time_s']:.2f}s ({r['decode_time_per_shot_s']*1e6:.1f} us/shot)  "
            f"sample={r['sample_time_s']:.2f}s"
        )
