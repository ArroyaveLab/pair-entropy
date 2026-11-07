"""Periodic lattices as neighbour tables, for tests and benchmarks."""
from __future__ import annotations

import numpy as np

__all__ = ["lattice"]


def lattice(kind: str, L: int):
    """Return (n_sites, neighbours) for a periodic lattice; neighbours is an (N, z) integer array.

    kind: "chain" (z = 2, L sites), "square" (z = 4, L x L), "bcc" (z = 8, 2 L^3 sites) or "fcc" (z = 12, 4 L^3 sites).
    """
    if kind == "chain":
        i = np.arange(L)
        return L, np.stack([(i - 1) % L, (i + 1) % L], 1)
    if kind == "square":
        idx = np.arange(L * L).reshape(L, L)
        offs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        return L * L, np.stack([np.roll(idx, (-a, -b), (0, 1)).ravel() for a, b in offs], 1)
    M = 2 * L
    g = np.indices((M, M, M)).reshape(3, -1).T
    if kind == "bcc":
        pts = g[(g % 2 == g[:, :1] % 2).all(1)]
        offs = np.array([(a, b, c) for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)])
    elif kind == "fcc":
        pts = g[g.sum(1) % 2 == 0]
        offs = np.array(sorted({(a, b, 0) for a in (-1, 1) for b in (-1, 1)}
                               | {(a, 0, b) for a in (-1, 1) for b in (-1, 1)}
                               | {(0, a, b) for a in (-1, 1) for b in (-1, 1)}))
    else:
        raise ValueError(f"unknown lattice {kind!r}")
    index = -np.ones((M, M, M), int)
    index[tuple(pts.T)] = np.arange(len(pts))
    nb = np.stack([index[tuple(((pts + o) % M).T)] for o in offs], 1)
    assert (nb >= 0).all()
    return len(pts), nb
