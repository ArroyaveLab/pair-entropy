"""Neighbour graphs of MD frames: partial radial distribution functions, first-minimum cutoffs and the chemical
term on the graphs, with block standard errors, cutoff cases and the label-permutation (ideal-mixing) check."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .estimator import BondCounter, adjacency_statistics

__all__ = ["distances", "neighbour_graph", "partial_rdfs", "first_minimum", "rdf_cutoffs", "pair_cutoff_matrix",
           "chemical_term", "block_se", "permutation_check"]


def distances(pos, box):
    """Minimum-image distance matrix (N, N) in an orthorhombic box with edges ``box`` (3,)."""
    d = pos[:, None, :] - pos[None, :, :]
    d -= box * np.round(d / box)
    return np.sqrt((d * d).sum(-1))


def neighbour_graph(pos, box, labels, rc):
    """Boolean adjacency: pairs closer than rc[label_a, label_b] (k x k cutoff matrix, A)."""
    r = distances(np.asarray(pos, float), box)
    np.fill_diagonal(r, np.inf)
    return r < rc[labels[:, None], labels[None, :]]


def partial_rdfs(traj, labels, k, stride=10, dr=0.025):
    """Partial RDFs g_ij (i <= j) and the species-blind total RDF over frames [::stride].
    Returns (r, {(i, j): g_ij}, g_total)."""
    pos, box = traj.positions, traj.box
    N = len(labels)
    edges = np.arange(0, box.min() / 2 + dr, dr)
    H = np.zeros((k, k, len(edges) - 1))
    Ht = np.zeros(len(edges) - 1)
    iu = np.triu_indices(N, 1)
    a, b = np.minimum(labels[iu[0]], labels[iu[1]]), np.maximum(labels[iu[0]], labels[iu[1]])
    nf = 0
    for f in range(0, pos.shape[0], stride):
        r = distances(pos[f], box)[iu]
        Ht += np.histogram(r, edges)[0]
        for i in range(k):
            for j in range(i, k):
                m = (a == i) & (b == j)
                if m.any():
                    H[i, j] += np.histogram(r[m], edges)[0]
        nf += 1
    rc = 0.5 * (edges[1:] + edges[:-1])
    shell = 4 * np.pi * rc ** 2 * dr
    V = float(np.prod(box))
    nsp = np.bincount(labels, minlength=k)
    g = {}
    for i in range(k):
        for j in range(i, k):
            npairs = nsp[i] * (nsp[j] - (i == j)) / (2 if i == j else 1)
            if npairs:
                g[(i, j)] = H[i, j] / nf / (npairs * shell / V)
    gt = Ht / nf / (N * (N - 1) / 2 * shell / V)
    return rc, g, gt


def first_minimum(r, g, window=2.0, smooth=5):
    """First minimum after the first (highest) peak of g(r): ``smooth``-bin running mean, searched within ``window`` A
    of the peak. Returns (r_min, r_peak)."""
    gs = np.convolve(g, np.ones(smooth) / smooth, mode="same")
    ip = int(np.argmax(gs))
    idx = np.where((r > r[ip]) & (r < r[ip] + window))[0]
    if not len(idx):
        raise ValueError("no first minimum found within the window")
    return float(r[idx[np.argmin(gs[idx])]]), float(r[ip])


def rdf_cutoffs(traj, species: Sequence[str], stride=10, dr=0.025):
    """First minima of the partial RDFs (k x k matrix) and of the total RDF. Species pairs absent from the state get
    the total-RDF minimum (they form no bonds, so it is never used)."""
    k = len(species)
    labels = traj.labels(species)
    r, g, gt = partial_rdfs(traj, labels, k, stride=stride, dr=dr)
    rcm = np.full((k, k), np.nan)
    peaks = {}
    for (i, j), gij in g.items():
        m, p = first_minimum(r, gij)
        rcm[i, j] = rcm[j, i] = m
        peaks[f"{species[i]}-{species[j]}"] = {"first_peak_A": p, "first_min_A": m}
    mt, pt = first_minimum(r, gt)
    rcm = np.where(np.isnan(rcm), mt, rcm)
    return {"rc_partial_A": rcm, "rc_total_A": mt, "total_first_peak_A": pt, "partials": peaks}


def pair_cutoff_matrix(rc, k):
    """A k x k cutoff matrix from a scalar or a matrix."""
    rc = np.asarray(rc, float)
    return np.full((k, k), float(rc)) if rc.ndim == 0 else rc.reshape(k, k)


def block_se(vals):
    """Standard error of the mean over blocks."""
    vals = np.asarray(vals, float)
    return float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else float("nan")


def _frames(traj, stride):
    return range(0, traj.nframes, stride)


def chemical_term(traj, species, rc, stride=1, nblock=5, rng=None):
    """Pool bond counts over the frames of ``traj`` with cutoff matrix ``rc`` (A).

    With ``rng`` (a numpy Generator), the species labels are permuted at random on every frame's neighbour graph,
    which is built with the true labels; this is the ideal-mixing check. Returns (total BondCounter, block counters).
    """
    k = len(species)
    labels = traj.labels(species)
    rc = pair_cutoff_matrix(rc, k)
    frames = list(_frames(traj, stride))
    nb = max(1, min(nblock, len(frames)))
    tot = BondCounter(k)
    blocks = [BondCounter(k) for _ in range(nb)]
    for n, f in enumerate(frames):
        adj = neighbour_graph(traj.positions[f], traj.box, labels, rc)
        lb = labels if rng is None else rng.permutation(labels)
        st = adjacency_statistics(lb, adj, k)
        tot.add(*st)
        blocks[n * nb // len(frames)].add(*st)
    return tot, blocks


def permutation_check(traj, species, rc, stride=1, nblock=5, seed=0):
    """Ideal-mixing check: excess chemical entropy (J/mol/K) of randomly permuted labels on the MD neighbour graphs.
    Returns a dict with the pooled value and its block standard error."""
    tot, blocks = chemical_term(traj, species, rc, stride, nblock, rng=np.random.default_rng(seed))
    e = tot.entropy()
    return {"S_chem_excess_permuted": e.excess, "block_SE": block_se([b.entropy().excess for b in blocks]),
            "S_ideal": e.ideal, "zbar": tot.zbar, "bonds_per_pair": tot.bonds_per_pair(species)}
