"""Small, independent revision-5 validator; no SCF loop or implicit retry.

validate(ctx, mol, W, sizes, counts, candidate) returns JSON-safe diagnostics.
ctx supplies S (physical AO overlap), guard(), record(event, **fields),
save_npz(name, **arrays), fock(mol, P, label, tol=None) -> F/J/K/h/EN,
and map_fock(W, S, F, sizes, counts, reference_occ) -> C_occ/P_occ/....
candidate supplies P_F, P_occ, C_occ (actual AO occupied columns per group),
energy, converged and history.  P is spin-summed.  Root selection belongs to
ctx.map_fock, never to an energy-sorted slice in this validator.  Physical
determinant qualification is deliberately separate from the source gate.
"""

from __future__ import annotations

import numpy as np


def _n2(a):
    return float(np.linalg.norm(a, 2)) if a.size else 0.0


def _safe(x):
    if isinstance(x, dict):
        return {str(k): _safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_safe(v) for v in x]
    if isinstance(x, np.ndarray):
        return _safe(x.tolist())
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return float(x) if np.isfinite(x) else None
    return x


def _matrix(x, name, shape=None):
    a = np.asarray(x)
    if a.dtype.kind not in "fciub":
        raise ValueError(f"{name}: numeric array required")
    a = np.asarray(a, dtype=complex if np.iscomplexobj(a) else float)
    if a.ndim != 2 or (shape is not None and a.shape != shape):
        raise ValueError(f"{name}: wrong shape {a.shape}, expected {shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: nonfinite data")
    return a


def _hermitian(a, name):
    defect = _n2(a - a.conj().T)
    if defect > 1e-10 * max(1.0, _n2(a)):
        raise ValueError(f"{name}: uncontrolled Hermitian defect {defect}")
    # Only roundoff symmetrization after the explicit defect gate.
    return (a + a.conj().T) * 0.5


def _complement(u):
    n, k = u.shape
    if not k:
        return np.eye(n, dtype=u.dtype)
    return np.linalg.qr(u, mode="complete")[0][:, k:]


def _energy_confirmed(history):
    rows = list(history or [])
    if len(rows) >= 3 and all("energy_change" in r for r in rows[-3:]):
        # Accept signed deltas, but confirmation depends on their magnitude.
        changes = np.abs([float(r["energy_change"]) for r in rows[-3:]])
    elif len(rows) >= 4 and all("energy" in r for r in rows[-4:]):
        changes = np.abs(np.diff([float(r["energy"]) for r in rows[-4:]]))
    else:
        return False
    return bool(np.all(np.isfinite(changes)) and np.max(changes) <= 1e-10)


def validate(ctx, mol, W, sizes, counts, candidate):
    """Validate one saved candidate, spending at most three new Fock builds.

    The two tight builds occur only after source snapshot/fresh-mapping gates.
    The result does not claim BUILD_PASS or full chemical IDENTITY_PASS; the
    caller combines those independent gates into SOURCE_BASELINE_PASS.
    No missing value is substituted with zero and no density is repaired.
    """
    raw = {}
    out = {
        "schema_version": "ful.minimum-validation/1.0",
        "status": "UNKNOWN", "numeric_pass": False,
        "original_c1": {"status": "UNKNOWN", "occupation_origin": "IMPOSED_FIXED_OCCUPATION"},
        "original_c2": {"status": "UNKNOWN"},
        "physical_determinant": {"status": "UNKNOWN"},
        "groups": [], "fock_builds": 0,
        "estimate_kind": "NUMERICAL_RELIABILITY_ESTIMATE",
        "original_overlap_integral_precision": {"value": None, "status": "UNASSESSED"},
        "certified_bound": None, "exact_nonlinear_root_distance_certified": False,
        "scope": "saved finite numerical model; construction and chemical identity assessed separately",
    }

    def finish():
        safe = _safe(out)
        ctx.save_npz("validation_arrays.npz", **raw)
        ctx.record("validation_complete", status=safe["status"],
                   numeric_pass=safe["numeric_pass"],
                   original_c2=safe["original_c2"]["status"],
                   fock_builds=safe["fock_builds"])
        return safe

    def fock(p, label, tol):
        ctx.guard()
        b = ctx.fock(mol, p, label, tol=tol)
        out["fock_builds"] += 1
        r = {}
        for key in ("F", "J", "K", "h"):
            r[key] = _matrix(b[key], f"{label}/{key}", (n, n))
            raw[f"{label}_{key}"] = r[key]
            r[key] = _hermitian(r[key], f"{label}/{key}")
        r["EN"] = float(b["EN"])
        if not np.isfinite(r["EN"]):
            raise ValueError("nonfinite nuclear energy")
        raw[f"{label}_EN"] = np.array(r["EN"])
        return r

    try:
        ctx.guard()
        required=('P_F','P_occ','C_occ','energy','converged','history')
        missing=[key for key in required if key not in candidate]
        if missing:
            raise ValueError('candidate missing required fields: '+','.join(missing))
        if not np.isfinite(float(candidate['energy'])):
            raise ValueError('candidate energy must be finite')
        W = _matrix(W, "W")
        n = W.shape[0]
        raw["W"] = W
        if W.shape != (n, n):
            raise ValueError("complete square W required by the Gamma_c contract")
        if any(isinstance(x, (bool, np.bool_)) or not float(x).is_integer()
               for x in [*sizes, *counts]):
            raise ValueError("integer sizes/counts required")
        sizes, counts = [int(x) for x in sizes], [int(x) for x in counts]
        if (len(sizes) != len(counts) or sum(sizes) != n
                or any(s <= 0 or k < 0 or k > s for s, k in zip(sizes, counts))):
            raise ValueError("invalid exhaustive group capacities")
        raw["group_sizes"], raw["occupied_counts"] = np.array(sizes), np.array(counts)
        out["metric_diagnostics"] = {}

        def metric_gate(a, name):
            diagonal = a.diagonal().real
            if np.any(diagonal <= 0):
                raise ValueError(f"{name}: nonpositive metric diagonal")
            normalized = a / np.sqrt(np.outer(diagonal, diagonal))
            eigenvalues = np.linalg.eigvalsh(normalized)
            relative = float(eigenvalues[0] / eigenvalues[-1])
            out["metric_diagnostics"][name] = {
                "normalized_relative_min_eigenvalue": relative,
                "normalized_condition_number": float(eigenvalues[-1] / eigenvalues[0]) if eigenvalues[0] > 0 else None,
                "condition_warning": bool(relative < 1e-8)}
            if relative <= 1e-12:
                raise ValueError(f"{name}: metric rank unresolved")
            return np.linalg.cholesky(a)

        s_input = getattr(ctx, "S", None)
        if s_input is None:
            ctx.record("validation_overlap_integral", operation="int1e_ovlp")
            s_input = mol.intor_symmetric("int1e_ovlp")
        S = _hermitian(_matrix(s_input, "S_AO", (n, n)), "S_AO")
        raw["S_AO"] = S
        la = metric_gate(S, "S_AO")
        SL = _hermitian(W.conj().T @ S @ W, "S_LFMO")
        metric_gate(SL, "S_LFMO")  # Complete-space rank; no implicit truncation.
        raw["S_LFMO"] = SL
        offsets = np.cumsum([0, *sizes])
        SC = np.zeros_like(SL)
        blocks = []
        for g, (s, k) in enumerate(zip(sizes, counts)):
            r = slice(offsets[g], offsets[g + 1])
            sg = SL[r, r]
            l = metric_gate(sg, f"group_{g}")
            x = np.linalg.solve(l.conj().T, np.eye(s))
            SC[r, r] = sg
            blocks.append({"r": r, "S": sg, "L": l, "X": x})
        lc = np.linalg.cholesky(SC)
        raw["S_cond"] = SC

        def gamma(p):
            a = np.linalg.solve(W, p)
            pw = np.linalg.solve(W, a.conj().T).conj().T
            return lc.conj().T @ pw @ lc

        def fp(pout, pin):
            numerator = float(np.linalg.norm(gamma(pout - pin), "fro"))
            denominator = max(float(np.linalg.norm(gamma(pout), "fro")), 1.0)
            return {"numerator": numerator, "denominator": denominator,
                    "relative": numerator / denominator}

        PF = _hermitian(_matrix(candidate["P_F"], "P_F", (n, n)), "P_F")
        P = _hermitian(_matrix(candidate["P_occ"], "P_occ", (n, n)), "P_occ")
        raw["P_F"], raw["P_occ"] = PF, P
        occ = list(candidate["C_occ"])
        if len(occ) != len(sizes):
            raise ValueError("actual C_occ required for every group")
        rebuilt = np.zeros_like(P, dtype=np.result_type(P, W, *[np.asarray(c).dtype for c in occ]))
        for g, (s, k, b) in enumerate(zip(sizes, counts, blocks)):
            c = _matrix(occ[g], f"C_occ_{g}", (n, k))
            raw[f"C_occ_{g}"] = c
            t = np.linalg.solve(W, c)
            og = t[b["r"], :]
            outside = t.copy()
            outside[b["r"], :] = 0
            if np.max(np.abs(outside), initial=0.0) > 1e-10:
                raise ValueError(f"group {g}: occupied columns escape their source group")
            u = b["L"].conj().T @ og
            gram_error = _n2(u.conj().T @ u - np.eye(k))
            if gram_error > 1e-10:
                raise ValueError(f"group {g}: actual occupied Gram fails, no density repair")
            q = _complement(u)
            qerr = max(_n2(q.conj().T @ q - np.eye(s - k)), _n2(q.conj().T @ u))
            if qerr > 1e-10:
                raise ValueError(f"group {g}: occupied-complement orthogonality fails")
            b.update(O=og, U=u, Q=q)
            raw[f"U_{g}"], raw[f"Q_{g}"] = u, q
            rebuilt += 2.0 * c @ c.conj().T
            out["groups"].append({"group": g, "size": s, "n_occupied": k,
                                  "occupied_gram_error": gram_error,
                                  "complement_error": qerr, "status": "UNKNOWN",
                                  "eta_int": None, "eta_met": None,
                                  "eta_eig": None, "eta_inv": None, "tau": None})
        raw["P_rebuilt"] = rebuilt
        out["density_reconstruction"] = fp(rebuilt, P)
        if out["density_reconstruction"]["relative"] > 1e-10:
            raise ValueError("saved P_occ disagrees with actual occupied source density")

        physical = _hermitian(la.conj().T @ P @ la, "physical density")
        natural = np.linalg.eigvalsh(physical)
        trace = float(np.trace(P @ S).real)
        expected = int(mol.nelectron)
        if 2 * sum(counts) != expected:
            raise ValueError("group electron ledger differs from the molecule")
        trace_error = abs(trace - expected)
        idem = float(np.linalg.norm(physical @ physical - 2 * physical, "fro")
                     / max(1.0, np.linalg.norm(2 * physical, "fro")))
        natural_distance = float(np.max(np.minimum(abs(natural), abs(natural - 2)), initial=0))
        raw["physical_natural_occupations"] = natural
        out["physical_determinant"] = {
            "status": "PASS" if trace_error <= 1e-6 and idem <= 1e-8 and natural_distance <= 1e-6 else "FAIL",
            "electron_trace": trace, "expected_electrons": expected,
            "electron_trace_error": trace_error, "idempotency_relative": idem,
            "natural_occupation_distance": natural_distance}
        out["original_c1"]["status"] = "PASS" if sizes[-1] == 4 and counts[-1] == 2 else "NOT_APPLICABLE_TO_EXTENDED_MODEL"
        out["energy_three_confirmations"] = _energy_confirmed(candidate.get("history"))
        out["solver_converged"] = bool(candidate.get("converged", False))
        if not out["solver_converged"] or not out["energy_three_confirmations"] or trace_error > 1e-6:
            out["reason"] = "source candidate not yet eligible for expensive fresh/precision verification"
            return finish()

        baseline_tol = float(getattr(ctx, "t", 1e-10))
        if not np.isfinite(baseline_tol) or baseline_tol <= 0:
            raise ValueError("invalid frozen integral screening threshold")
        out["integral_screen_threshold"] = baseline_tol
        out["integral_control"] = getattr(ctx, "integral_control", "direct_scf_tol")
        base = fock(P, "fresh_t", baseline_tol)
        raw["fresh_F_LFMO"] = W.conj().T @ base["F"] @ W
        raw["fresh_F_cond"] = np.zeros_like(raw["fresh_F_LFMO"])
        for b in blocks:
            raw["fresh_F_cond"][b["r"], b["r"]] = raw["fresh_F_LFMO"][b["r"], b["r"]]
        fc = base["F"] - base["h"] - base["J"] + 0.5 * base["K"]
        e1 = float(np.trace(P @ base["h"]).real)
        ej = 0.5 * float(np.trace(P @ base["J"]).real)
        ek = -0.25 * float(np.trace(P @ base["K"]).real)
        energy = e1 + ej + ek + base["EN"]
        alternate = 0.5 * float(np.trace(P @ (base["h"] + base["F"])).real) + base["EN"]
        saved_energy = float(candidate["energy"])
        if not np.isfinite(saved_energy):
            raise ValueError("nonfinite saved candidate energy")
        out["energy"] = {"one_electron": e1, "coulomb": ej, "exchange": ek,
                         "nuclear": base["EN"], "total": energy,
                         "half_trace_total": alternate,
                         "assembly_error": abs(energy - alternate),
                         "candidate_energy_error": abs(energy - saved_energy),
                         "fock_identity_error": _n2(fc),
                         "evidence": "independent array contractions sharing the same J/K"}
        ctx.guard()
        mapped = ctx.map_fock(W, S, base["F"], sizes, counts, occ)
        PN = _hermitian(_matrix(mapped["P_occ"], "P_next", (n, n)), "P_next")
        raw["P_next"] = PN
        if "C" in mapped:
            raw["C_next"] = np.asarray(mapped["C"])
        next_occ = list(mapped["C_occ"])
        if len(next_occ) != len(counts):
            raise ValueError("fresh map omitted actual occupied groups")
        next_rebuilt = np.zeros_like(rebuilt)
        for g, (c, k) in enumerate(zip(next_occ, counts)):
            c = _matrix(c, f"C_occ_next_{g}", (n, k))
            raw[f"C_occ_next_{g}"] = c
            next_rebuilt += 2.0 * c @ c.conj().T
        raw["P_next_rebuilt"] = next_rebuilt
        out["next_density_reconstruction"] = fp(next_rebuilt, PN)
        if out["next_density_reconstruction"]["relative"] > 1e-10:
            raise ValueError("fresh mapped P_next disagrees with mapped occupied columns")
        for key in ("energies", "permutations"):
            if key in mapped:
                for g, value in enumerate(mapped[key]):
                    raw[f"next_{key}_{g}"] = np.asarray(value)
        root_min_sv = float(mapped["root_min_sv"])
        out["root_min_sv"] = root_min_sv
        if not np.isfinite(root_min_sv) or root_min_sv < float(getattr(ctx, "root_overlap_min", 0.5)):
            raise ValueError("fresh root overlap below frozen identity gate")
        if mapped.get("root_identity_resolved") is False:
            raise ValueError("fresh root assignment unresolved")
        out["fixed_point_occ_F"] = fp(P, PF)
        out["fixed_point_next_occ"] = fp(PN, P)
        energy_ok = abs(energy - alternate) <= 1e-8 and abs(energy - saved_energy) <= 1e-8 and _n2(fc) <= 1e-10
        invariant = True
        for g, (k, b, row) in enumerate(zip(counts, blocks, out["groups"])):
            f = W[:, b["r"]].conj().T @ base["F"] @ W[:, b["r"]]
            h = _hermitian(b["X"].conj().T @ f @ b["X"], f"H_{g}")
            av = _hermitian(b["Q"].conj().T @ h @ b["Q"], f"Av_{g}")
            lam, z = np.linalg.eigh(av)
            if z.size:
                uz, _, vhz = np.linalg.svd(z, full_matrices=False)
                zo = uz @ vhz
            else:
                zo = z
            coupling = b["Q"].conj().T @ h @ b["U"]
            inv = _n2(coupling)
            invmax = float(np.max(abs(coupling), initial=0))
            eB = _n2(b["X"].conj().T @ b["S"] @ b["X"] - np.eye(len(h)))
            es, vs = np.linalg.eigh(b["S"])
            sx = (vs * (es ** -0.5)) @ vs.conj().T
            sr = (vs * (es ** 0.5)) @ vs.conj().T
            us = sr @ b["O"]
            qs = _complement(us)
            white_err = max(_n2(us.conj().T @ us - np.eye(k)),
                            _n2(qs.conj().T @ us),
                            _n2(qs.conj().T @ qs - np.eye(qs.shape[1])),
                            _n2(z.conj().T @ z - np.eye(z.shape[1])),
                            _n2(zo.conj().T @ zo - np.eye(zo.shape[1])))
            other = np.linalg.eigvalsh(_hermitian(qs.conj().T @ sx @ f @ sx @ qs, f"symmetric_Av_{g}"))
            delta = float(np.max(abs(lam - other), initial=0))
            met = delta + _n2(h) * eB / (1 - eB) if eB < 1 else float("inf")
            row.update(virtual_energies=lam.tolist(), eta_inv=inv, inv_max_abs=invmax,
                       eta_eig=_n2(av - (zo * lam) @ zo.conj().T),
                       eta_met=met, e_B=eB, delta_white=delta,
                       whitening_orthogonality_error=white_err,
                       virtual_dimension=len(lam))
            b.update(H=h, eigenvalues=lam)
            raw[f"H_{g}"], raw[f"Av_{g}"], raw[f"virtual_energies_{g}"] = h, av, lam
            raw[f"virtual_eigenvectors_{g}"], raw[f"virtual_eigenvectors_polar_{g}"] = z, zo
            raw[f"symmetric_virtual_energies_{g}"] = other
            raw[f"occupied_virtual_coupling_{g}"] = coupling
            invariant &= inv <= 1e-7 and invmax <= 1e-7 and eB <= 1e-10 and white_err <= 1e-10
        fp_ok = out["fixed_point_occ_F"]["relative"] <= 1e-8 and out["fixed_point_next_occ"]["relative"] <= 1e-8
        if not invariant or not fp_ok or not energy_ok:
            out["reason"] = "fresh invariance/fixed-point/energy gate failed; tight Fock builds skipped"
            return finish()

        tight1 = fock(P, "tight_01t", baseline_tol * 0.1)
        tight2 = fock(P, "tight_001t", baseline_tol * 0.01)
        stable = True
        for g, (b, row) in enumerate(zip(blocks, out["groups"])):
            wg, x = W[:, b["r"]], b["X"]
            diff = x.conj().T @ wg.conj().T @ (tight2["F"] - base["F"]) @ wg @ x
            last = x.conj().T @ wg.conj().T @ (tight2["F"] - tight1["F"]) @ wg @ x
            eta, stability = _n2(diff), _n2(last)
            tau = 1e-7 + 2 * eta + 2 * row["eta_met"] + row["eta_eig"] + row["eta_inv"]
            lam = b["eigenvalues"]
            ok = stability <= 1e-8
            stable &= ok
            status = ("UNKNOWN" if not ok else "VACUOUS_NO_VIRTUALS" if not len(lam)
                      else "FAIL" if np.any(lam < -tau)
                      else "PASS" if np.all(lam > tau) else "UNKNOWN")
            row.update(eta_int=eta, integral_last_difference=stability,
                       integral_stable=ok, tau=tau, status=status,
                       n_reliable_negative=int(np.count_nonzero(lam < -tau)),
                       n_near_zero=int(np.count_nonzero(abs(lam) <= tau)))
            raw[f"integral_difference_{g}"], raw[f"integral_last_difference_{g}"] = diff, last
        statuses = [r["status"] for r in out["groups"]]
        c2 = ("FAIL" if "FAIL" in statuses else "PASS"
              if all(s in ("PASS", "VACUOUS_NO_VIRTUALS") for s in statuses) else "UNKNOWN")
        out["original_c2"] = {"status": c2, "units": "Eh", "relative_band": 0.0}
        out["numeric_pass"] = bool(stable)
        out["status"] = "NUMERICALLY_CONVERGED" if stable else "UNKNOWN"
        if not stable:
            out["reason"] = "fixed-density integral precision comparison not stable in every group"
        return finish()
    except Exception as exc:
        out["reason"] = f"{type(exc).__name__}: {exc}"
        return finish()
