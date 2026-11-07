"""Pair (quasi-chemical) entropy of species labels on a neighbour graph.

For a graph on which every site carries a species label, let x_i be the site fraction of species i, y_ij the
fraction of ordered bonds with species i at one end and j at the other (a symmetric k x k matrix that sums to one),
w_i = sum_j y_ij the bond-end fraction of species i and zbar the mean coordination. The pair estimator is

    S_chem / R = -sum_i x_i ln x_i - (zbar/2) sum_ij y_ij ln[ y_ij / (w_i w_j) ],

(Guggenheim 1944; Kikuchi 1951). The second term is -(zbar/2) times the mutual information I of the species at
the two ends of a bond, so that the excess chemical entropy is S_chem,ex = -R (zbar/2) I <= 0, with equality for
random labels on any graph.

Units: J/mol/K (per mole of sites) unless stated otherwise.
"""
from __future__ import annotations

from typing import NamedTuple, Sequence

import numpy as np

R = 8.314462618  # J/mol/K

__all__ = ["R", "PairEntropy", "shannon", "ideal_mixing", "mutual_information", "pair_entropy",
           "pair_entropy_x_reference", "bethe_quasichemical", "bond_statistics", "adjacency_statistics",
           "BondCounter"]


class PairEntropy(NamedTuple):
    """Chemical entropy, its ideal-mixing part and the excess, J/mol/K."""

    total: float
    ideal: float
    excess: float


def shannon(p) -> float:
    """Shannon entropy -sum p ln p (nats) of a probability vector or array; zero entries are skipped."""
    p = np.asarray(p, float).ravel()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def ideal_mixing(x) -> float:
    """Ideal entropy of mixing -R sum x_i ln x_i, J/mol/K. ``x`` is a composition vector, or x_B of a binary."""
    x = np.atleast_1d(np.asarray(x, float))
    if x.size == 1:
        x = np.array([1.0 - x[0], x[0]])
    return R * shannon(x)


def _check_bonds(y):
    y = np.asarray(y, float)
    if y.ndim != 2 or y.shape[0] != y.shape[1]:
        raise ValueError("y must be a square k x k matrix of ordered bond fractions")
    if np.any(y < 0):
        raise ValueError("bond fractions must be non-negative")
    s = y.sum()
    if not np.isclose(s, 1.0, rtol=0, atol=1e-9):
        raise ValueError(f"bond fractions must sum to 1 (sum = {s})")
    return y


def mutual_information(y) -> float:
    """Mutual information (nats) of the species at the two ends of a bond, from ordered bond fractions y_ij."""
    y = _check_bonds(y)
    w = y.sum(1)
    m = y > 0
    return float((y[m] * np.log(y[m] / np.outer(w, w)[m])).sum())


def pair_entropy(x: Sequence[float], y, zbar: float) -> PairEntropy:
    """Pair (quasi-chemical) entropy of the labels.

    x: site fractions (k,); y: ordered bond fractions (k, k), symmetric, summing to one; zbar: mean coordination.
    Returns PairEntropy(total, ideal, excess) in J/mol/K.
    """
    x = np.asarray(x, float)
    s_id = R * shannon(x)
    ex = -R * zbar / 2.0 * mutual_information(y)
    return PairEntropy(s_id + ex, s_id, ex)


def pair_entropy_x_reference(x: Sequence[float], y, zbar: float) -> PairEntropy:
    """Sensitivity variant with the reference x_i x_j in place of the bond-end fractions w_i w_j.

    Identical to :func:`pair_entropy` on regular graphs. On a graph whose coordination depends on the species (a
    liquid) it also books the coupling between species and coordination number as chemical, so it is not zero for
    random labels in general; it is reported as a sensitivity case only.
    """
    x = np.asarray(x, float)
    y = _check_bonds(y)
    m = y > 0
    kl = float((y[m] * np.log(y[m] / np.outer(x, x)[m])).sum())
    s_id = R * shannon(x)
    return PairEntropy(s_id - R * zbar / 2.0 * kl, s_id, -R * zbar / 2.0 * kl)


def bethe_quasichemical(z: float, x_B: float, alpha: float) -> PairEntropy:
    """Pair estimator for a binary A-B with site fraction x_B and Warren-Cowley parameter alpha on a lattice of
    coordination z (the quasi-chemical bond fractions y_AB = x_A x_B (1 - alpha))."""
    xA = 1.0 - x_B
    yAB = xA * x_B * (1.0 - alpha)
    y = np.array([[xA - yAB, yAB], [yAB, x_B - yAB]])
    if np.any(y < -1e-15):
        raise ValueError(f"alpha = {alpha} is not attainable at x_B = {x_B}")
    return pair_entropy([xA, x_B], np.clip(y, 0, None), z)


# ---------------------------------------------------------------------------------------------- counting
def _onehot(labels, k):
    return np.eye(k, dtype=np.int64)[labels]


def bond_statistics(labels, neighbours, k: int | None = None):
    """Neighbour counts by species.

    labels: (N,) integer species 0..k-1; neighbours: (N, z) integer array, or a list of N neighbour-index arrays.
    Returns (labels, degree, counts) with counts[a, s] the number of species-s neighbours of site a.
    """
    labels = np.asarray(labels, int)
    k = int(labels.max()) + 1 if k is None else k
    oh = _onehot(labels, k)
    if isinstance(neighbours, np.ndarray) and neighbours.ndim == 2:
        counts = oh[neighbours].sum(1)
    else:
        counts = np.array([oh[np.asarray(a, int)].sum(0) if len(a) else np.zeros(k, np.int64) for a in neighbours])
    return labels, counts.sum(1), counts


def adjacency_statistics(labels, adjacency, k: int | None = None):
    """As :func:`bond_statistics`, from a symmetric boolean adjacency matrix (N, N)."""
    labels = np.asarray(labels, int)
    k = int(labels.max()) + 1 if k is None else k
    counts = np.asarray(adjacency).astype(np.int64) @ _onehot(labels, k)
    return labels, counts.sum(1), counts


class BondCounter:
    """Pools ordered-bond counts (k x k), site counts and coordination over frames or samples."""

    def __init__(self, k: int = 2):
        self.k = k
        self.bonds = np.zeros((k, k))      # bonds[i, j]: ordered bonds with i at the centre end and j at the other
        self.sites = np.zeros(k)
        self.degree_sum = 0.0

    def add(self, labels, degree, counts):
        labels = np.asarray(labels, int)
        for s in range(self.k):
            m = labels == s
            if m.any():
                self.bonds[s] += counts[m].sum(0)
                self.sites[s] += m.sum()
        self.degree_sum += float(np.sum(degree))
        return self

    @property
    def x(self):
        return self.sites / self.sites.sum()

    @property
    def y(self):
        y = self.bonds / self.bonds.sum()
        return 0.5 * (y + y.T)   # exact for an undirected graph; guards against rounding

    @property
    def zbar(self):
        return self.degree_sum / self.sites.sum()

    def entropy(self) -> PairEntropy:
        return pair_entropy(self.x, self.y, self.zbar)

    def entropy_x_reference(self) -> PairEntropy:
        return pair_entropy_x_reference(self.x, self.y, self.zbar)

    def bonds_per_pair(self, names: Sequence[str] | None = None) -> dict:
        """Undirected bonds sampled for each species pair (summed over what was added)."""
        names = list(names) if names is not None else [str(i) for i in range(self.k)]
        return {f"{names[i]}-{names[j]}": float(self.bonds[i, j] if i != j else self.bonds[i, i] / 2)
                for i in range(self.k) for j in range(i, self.k)}
