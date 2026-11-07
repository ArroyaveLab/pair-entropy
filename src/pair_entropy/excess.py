"""Excess entropy of alloy liquids, S_ex = S_vib,ex + S_chem,ex, from trajectories of the alloys and of the pure
liquid of every species at the same temperature, and its mapping to Redlich-Kister b_k parameters."""
from __future__ import annotations

import itertools
from typing import Sequence

import numpy as np

from .graph import block_se, chemical_term, pair_cutoff_matrix, permutation_check, rdf_cutoffs
from .vibrational import kinetic_temperature, vibrational_entropy

__all__ = ["chemical_excess", "excess_entropy", "redlich_kister_design", "fit_redlich_kister"]

MIN_PAIR_BONDS = 1000   # warn when a species pair would have fewer sampled bonds under random mixing


def _cutoff_cases(cut, k, cutoff, rc, rc_delta):
    if rc is not None:
        base = pair_cutoff_matrix(rc, k)
        return {"fixed": base, "fixed_minus_delta": base - rc_delta, "fixed_plus_delta": base + rc_delta}
    par = np.asarray(cut["rc_partial_A"], float)
    tot = np.full((k, k), cut["rc_total_A"])
    base, other = (par, tot) if cutoff == "partial" else (tot, par)
    oname = "total" if cutoff == "partial" else "partial"
    return {f"{cutoff}_rdf_minimum": base, f"{cutoff}_rdf_minimum_minus_delta": base - rc_delta,
            f"{cutoff}_rdf_minimum_plus_delta": base + rc_delta, f"{oname}_rdf_minimum": other}


def _pm(M, species):
    k = len(species)
    return {f"{species[i]}-{species[j]}": float(M[i][j]) for i in range(k) for j in range(i, k)}


def chemical_excess(traj, species: Sequence[str], cutoff="partial", rc=None, rc_delta=0.1, stride=1, nblock=5,
                    rdf_stride=10, seed=0):
    """Chemical term of one state on its MD neighbour graphs.

    cutoff: "partial" (first minima of the partial RDFs, default) or "total" (first minimum of the total RDF); rc: a
    fixed cutoff (A, scalar or k x k) that overrides both. The cutoff cases are the base, base -/+ rc_delta and the
    other RDF convention (or fixed -/+ rc_delta); their largest change from the base is the cutoff spread.
    Returns a dict (J/mol/K) including the label-permutation check on the base graphs.
    """
    species = list(species)
    k = len(species)
    cut = rdf_cutoffs(traj, species, stride=rdf_stride) if rc is None else None
    cases = _cutoff_cases(cut, k, cutoff, rc, rc_delta)
    rows = {}
    for name, rcm in cases.items():
        tot, blocks = chemical_term(traj, species, rcm, stride, nblock)
        e = tot.entropy()
        rows[name] = {"rc_A": _pm(rcm, species), "S_chem_excess": e.excess,
                      "S_chem_excess_block_SE": block_se([b.entropy().excess for b in blocks]), "zbar": tot.zbar,
                      "_counter": tot, "_entropy": e}
    b0 = next(iter(rows))
    base = rows[b0]
    tot, e = base.pop("_counter"), base.pop("_entropy")
    for r in rows.values():
        r.pop("_counter", None)
        r.pop("_entropy", None)
    spread = max(abs(r["S_chem_excess"] - base["S_chem_excess"]) for r in rows.values())
    check = permutation_check(traj, species, cases[b0], stride, nblock, seed)
    out = {"x": dict(zip(species, tot.x.tolist())), "S_ideal": e.ideal, "S_chem": e.total,
           "S_chem_excess": e.excess, "S_chem_excess_block_SE": base["S_chem_excess_block_SE"],
           "S_chem_excess_cutoff_spread": float(spread),
           "S_chem_excess_u": float(np.hypot(base["S_chem_excess_block_SE"], spread)),
           "S_chem_excess_x_reference_variant": tot.entropy_x_reference().excess,
           "zbar": tot.zbar, "y_bonds": tot.y.tolist(), "bond_counts_ordered": tot.bonds.tolist(),
           "bonds_per_pair": tot.bonds_per_pair(species), "frames_in_graph": len(range(0, traj.nframes, stride)),
           "cutoff_case": b0, "cutoff_cases": rows, "permutation_check": check}
    if cut is not None:
        out["neighbour_cutoffs"] = {"rc_partial_A": _pm(cut["rc_partial_A"], species), "rc_total_A": cut["rc_total_A"],
                                    "total_first_peak_A": cut["total_first_peak_A"], "partials": cut["partials"]}
    return out


def redlich_kister_design(X, order):
    """Columns -x_i x_j (x_i - x_j)^k for every pair i < j and k = 0..order (no ternary terms)."""
    X = np.asarray(X, float)
    cols, names = [], []
    for i, j in itertools.combinations(range(X.shape[1]), 2):
        for k in range(order + 1):
            cols.append(-X[:, i] * X[:, j] * (X[:, i] - X[:, j]) ** k)
            names.append((i, j, k))
    return np.stack(cols, 1), names


def fit_redlich_kister(X, s_ex, cov, order):
    """Generalised least squares for S_ex = -sum_{i<j} x_i x_j sum_k b_k^ij (x_i - x_j)^k, the temperature
    derivative of a Redlich-Kister excess with L_k = a_k + b_k T. For a binary with x = x_B and species in
    alphabetical order, S_ex = -x (1 - x) sum_k b_k (1 - 2x)^k."""
    A, names = redlich_kister_design(X, order)
    Ci = np.linalg.inv(np.asarray(cov, float))
    covb = np.linalg.inv(A.T @ Ci @ A)
    s_ex = np.asarray(s_ex, float)
    b = covb @ A.T @ Ci @ s_ex
    res = s_ex - A @ b
    return {"order": order, "b": b.tolist(), "cov_b": covb.tolist(), "b_SE": np.sqrt(np.diag(covb)).tolist(),
            "columns": names, "chi2": float(res @ Ci @ res), "dof": int(len(s_ex) - len(b))}


def excess_entropy(alloys: dict, references: dict, T: float, dt_fs: float, cutoff="partial", rc=None, rc_delta=0.1,
                   tmax_ps=3.0, nblock=5, stride=1, rdf_stride=10, ideal_tol=0.05, seed=0, order=None):
    """Excess entropy of alloy liquids, J/mol-atom/K.

    alloys: {name: Trajectory}; references: {species: Trajectory of the pure liquid}; T: temperature (K) of every
    state; dt_fs: time between frames. S_ex = S_vib,ex + S_chem,ex; the excess electronic entropy is taken as
    zero (metallic liquids), an assumption stated in the result. Excess entropies are marked as held when the label-permutation check of any alloy exceeds
    ``ideal_tol``.
    """
    species = sorted(references)
    k = len(species)
    if k < 2:
        raise ValueError("give one pure-liquid reference per species (at least two)")
    warn = []
    for name, tr in references.items():
        if set(tr.symbols) != {name}:
            raise ValueError(f"reference {name!r} must be a pure liquid of {name}, found {tr.species()}")
    for name, tr in alloys.items():
        bad = set(tr.symbols) - set(species)
        if bad:
            raise ValueError(f"{name}: species {sorted(bad)} have no pure-liquid reference")
    states = {**{f"ref:{s}": t for s, t in references.items()}, **alloys}
    vib, tkin = {}, {}
    for name, tr in states.items():
        if tr.nframes // nblock - 1 < tmax_ps * 1000 / dt_fs:
            warn.append(f"{name}: block length {tr.nframes // nblock * dt_fs / 1000:g} ps is shorter than the VACF "
                        f"window {tmax_ps:g} ps; block SEs use a truncated window")
        vib[name] = vibrational_entropy(tr, dt_fs, T, tmax_ps, nblock)
        tkin[name] = kinetic_temperature(tr)
        if abs(tkin[name] - T) / T > 0.05:
            warn.append(f"{name}: mean kinetic temperature {tkin[name]:.1f} K differs from T = {T:g} K by > 5 %")
    rv = {s: vib[f"ref:{s}"] for s in species}
    out_alloys = {}
    ideal_ok = True
    for name, tr in alloys.items():
        lab = tr.labels(species)
        x = np.bincount(lab, minlength=k) / len(lab)
        v = vib[name]
        vex = v["S_vib"] - sum(x[i] * rv[s]["S_vib"] for i, s in enumerate(species))
        vse = float(np.sqrt(v["S_vib_block_SE"] ** 2 + sum((x[i] * rv[s]["S_vib_block_SE"]) ** 2
                                                           for i, s in enumerate(species))))
        ch = chemical_excess(tr, species, cutoff, rc, rc_delta, stride, nblock, rdf_stride, seed)
        pc = ch["permutation_check"]
        pc["passed"] = bool(abs(pc["S_chem_excess_permuted"]) < ideal_tol)
        ideal_ok &= pc["passed"]
        nb = sum(ch["bonds_per_pair"].values())
        for i, j in itertools.combinations_with_replacement(range(k), 2):
            pair = f"{species[i]}-{species[j]}"
            expect = nb * x[i] * x[j] * (1 if i == j else 2)
            if 0 < expect < MIN_PAIR_BONDS:
                warn.append(f"{name}: {pair} would have only {expect:.0f} sampled bonds under random mixing "
                            f"(< {MIN_PAIR_BONDS}; observed {ch['bonds_per_pair'][pair]:.0f})")
        out_alloys[name] = {"N": tr.natoms, "frames": tr.nframes, "T_kinetic_K": tkin[name],
                            "S_vib": v["S_vib"], "S_vib_block_SE": v["S_vib_block_SE"],
                            "S_vib_excess": float(vex), "S_vib_excess_SE": vse,
                            **ch,
                            "S_excess": float(vex + ch["S_chem_excess"]),
                            "S_excess_u": float(np.hypot(vse, ch["S_chem_excess_u"])),
                            "S_electronic_excess": 0.0,
                            "dos_integral_per_atom": v["dos_integral_per_atom"],
                            "negative_dos_weight_per_atom": v["negative_dos_weight_per_atom"]}
    for r in out_alloys.values():
        r["status"] = "computed" if ideal_ok else "held (label-permutation check failed)"
    if not ideal_ok:
        warn.append("label-permutation check FAILED: random labels on the MD neighbour graphs do not give "
                    "S_chem,ex ~ 0; excess entropies are held")
    # Redlich-Kister mapping with the covariance from shared references
    keys = list(out_alloys)
    X = np.array([[out_alloys[n]["x"][s] for s in species] for n in keys])
    sex = np.array([out_alloys[n]["S_excess"] for n in keys])
    cov = np.diag([out_alloys[n]["S_vib_block_SE"] ** 2 + out_alloys[n]["S_chem_excess_u"] ** 2 for n in keys])
    for i, s in enumerate(species):
        cov += np.outer(X[:, i], X[:, i]) * rv[s]["S_vib_block_SE"] ** 2
    npairs = k * (k - 1) // 2
    n = len(keys)
    od = order if order is not None else min(2, n // npairs - 1)
    rk = {"model": "S_ex = -sum_{i<j} x_i x_j sum_k b_k^{ij} (x_i - x_j)^k, from L_k = a_k + b_k T",
          "species_order": species, "compositions": X.tolist(), "S_excess": sex.tolist(), "cov_S_excess": cov.tolist()}
    if od < 0 or npairs * (od + 1) > n:
        rk["fit"] = None
        rk["note"] = (f"not determined: {n} composition(s) for {npairs} species pair(s); use S_excess and its "
                      "covariance directly")
    else:
        fit = fit_redlich_kister(X, sex, cov, od)
        fit["columns"] = [f"b_{kk}^{species[i]},{species[j]}" for i, j, kk in fit["columns"]]
        rk["fit"] = fit
    return {"units": "J/mol-atom/K", "T_K": T, "species_order": species,
            "permutation_check": {"criterion": f"|S_chem,ex(permuted labels)| < {ideal_tol} J/mol-atom/K",
                                  "passed": ideal_ok},
            "references": {s: {"source": references[s].source, **rv[s], "T_kinetic_K": tkin[f"ref:{s}"],
                               "N": references[s].natoms, "frames": references[s].nframes} for s in species},
            "alloys": out_alloys, "redlich_kister": rk,
            "assumptions": {"electronic": "excess electronic entropy taken as zero (metallic liquids)"},
            "uncertainty": "S_chem,ex u = quadrature(block SE, largest cutoff-case change); S_vib,ex SE = quadrature "
                           "of the block SEs of the alloy and the x-weighted references; covariance from shared "
                           "references; potential and method errors not included",
            "warnings": warn}
