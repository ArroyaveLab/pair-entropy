"""Random labels give ideal mixing: in the thermodynamic limit (machine precision), on exactly enumerated lattice
ensembles (machine precision), on sampled lattices and irregular graphs, and under label permutation."""
import itertools

import numpy as np
import pytest

import pair_entropy as pe
from pair_entropy import R, shannon

XS = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
rng = np.random.default_rng(1)


@pytest.mark.parametrize("z", [2, 4, 8, 12])
def test_random_labels_ideal_thermodynamic_limit_binary(z):
    """y_ij = x_i x_j: the pair estimator equals -R sum x ln x, and so does the lattice form (z/2) H(y) - (z-1) H(x)."""
    for x in XS:
        s = pe.bethe_quasichemical(z, x, 0.0)
        assert s.total == pytest.approx(pe.ideal_mixing(x), abs=1e-12)
        assert abs(s.excess) < 1e-12
        y = np.outer([1 - x, x], [1 - x, x])
        lattice_form = R * ((z / 2) * shannon(y) - (z - 1) * shannon([1 - x, x]))
        assert lattice_form == pytest.approx(pe.ideal_mixing(x), abs=1e-12)


@pytest.mark.parametrize("z", [4, 12])
@pytest.mark.parametrize("x", [(0.2, 0.3, 0.5), (1 / 3, 1 / 3, 1 / 3), (0.1, 0.2, 0.3, 0.4), (0.05, 0.15, 0.8)])
def test_random_labels_ideal_thermodynamic_limit_multicomponent(z, x):
    x = np.array(x)
    s = pe.pair_entropy(x, np.outer(x, x), z)
    assert s.total == pytest.approx(-R * (x * np.log(x)).sum(), abs=1e-12)
    assert abs(s.excess) < 1e-12


def _exact_enumeration(nb, xB):
    """All 2^N labellings of a small periodic lattice, i.i.d. with probability xB: probability-weighted bond
    fractions, and the entropy of the joint (labels, neighbour counts) configuration."""
    N, z = nb.shape
    probs, ybar = {}, np.zeros((2, 2))
    for bits in itertools.product((0, 1), repeat=N):
        lab = np.array(bits, int)
        p = xB ** lab.sum() * (1 - xB) ** (N - lab.sum())
        sig = tuple(lab) + tuple(lab[nb].sum(1))
        probs[sig] = probs.get(sig, 0) + p
        _, _, counts = pe.bond_statistics(lab, nb, 2)
        for s in (0, 1):
            ybar[s] += p * counts[lab == s].sum(0)
    return shannon(list(probs.values())), ybar / ybar.sum()


@pytest.mark.parametrize("graph", ["ring12", "square4x4"])
@pytest.mark.parametrize("xB", [0.1, 0.5])
def test_exact_enumeration_on_lattices_is_ideal(graph, xB):
    """Ring of 12 (z = 2) and 4 x 4 square torus (z = 4), every labelling enumerated: the pair estimator on the exact
    ensemble equals ideal mixing to machine precision, as does the entropy of the full configuration per site."""
    nb = pe.lattice("chain", 12)[1] if graph == "ring12" else pe.lattice("square", 4)[1]
    joint, y = _exact_enumeration(nb, xB)
    assert R * joint / nb.shape[0] == pytest.approx(pe.ideal_mixing(xB), abs=1e-10)
    s = pe.pair_entropy([1 - xB, xB], y, nb.shape[1])
    assert s.total == pytest.approx(pe.ideal_mixing(xB), abs=1e-10)
    assert abs(s.excess) < 1e-10


@pytest.mark.parametrize("kind,L", [("chain", 20000), ("square", 100), ("bcc", 12), ("fcc", 10)])
@pytest.mark.parametrize("x", [(0.1, 0.9), (0.5, 0.5), (0.2, 0.3, 0.5), (0.1, 0.2, 0.3, 0.4)])
def test_random_labels_sampled_lattices(kind, L, x):
    """Random labels on periodic z = 2, 4, 8, 12 lattices (k = 2, 3, 4), pooled samples: pair term = ideal to 0.01."""
    N, nb = pe.lattice(kind, L)
    k = len(x)
    bc = pe.BondCounter(k)
    for _ in range(3):
        bc.add(*pe.bond_statistics(rng.choice(k, size=N, p=x), nb, k))
    e = bc.entropy()
    assert abs(e.excess) < 0.01
    assert e.ideal == pytest.approx(pe.ideal_mixing(bc.x))


def _random_irregular_graph(N, dmin, dmax):
    deg = rng.integers(dmin, dmax + 1, N)
    stubs = np.repeat(np.arange(N), deg)
    stubs = stubs[: len(stubs) // 2 * 2]
    rng.shuffle(stubs)
    nbr = [[] for _ in range(N)]
    for a, b in zip(stubs[::2], stubs[1::2]):
        nbr[a].append(b)
        nbr[b].append(a)
    return [np.array(v, int) for v in nbr]


def test_random_labels_irregular_graph():
    """Random labels on an irregular graph (degrees 10-16, liquid-like), ternary: the bond-end reference keeps the
    pair term at ideal mixing."""
    nbr = _random_irregular_graph(4000, 10, 16)
    bc = pe.BondCounter(3)
    for _ in range(10):
        bc.add(*pe.bond_statistics(rng.choice(3, size=4000, p=(0.2, 0.3, 0.5)), nbr, 3))
    assert abs(bc.entropy().excess) < 0.01


def test_x_reference_variant_books_coordination_coupling():
    """When the coordination depends on the species, the x_i x_j-referenced variant is not ideal for random labels,
    while the w_i w_j (bond-end) estimator is."""
    N = 6000
    lab = rng.choice(2, size=N, p=(0.5, 0.5))
    # species 1 has twice the coordination of species 0 (stub counts), wired at random
    deg = np.where(lab == 1, 16, 8)
    stubs = np.repeat(np.arange(N), deg)
    rng.shuffle(stubs)
    nbr = [[] for _ in range(N)]
    for a, b in zip(stubs[::2], stubs[1::2]):
        nbr[a].append(b)
        nbr[b].append(a)
    bc = pe.BondCounter(2).add(*pe.bond_statistics(lab, [np.array(v, int) for v in nbr], 2))
    assert abs(bc.entropy().excess) < 0.01
    assert bc.entropy_x_reference().excess < -1.0


@pytest.mark.parametrize("kind,L", [("square", 60), ("chain", 3000)])
def test_label_permutation_on_ordered_lattice_gives_zero(kind, L):
    """An ordered labelling has a large negative excess; the same labels permuted at random on the same graph give
    ~0 (the ideal-mixing check)."""
    N, nb = pe.lattice(kind, L)
    if kind == "square":
        i, j = np.divmod(np.arange(N), L)
        lab = (i + j) % 2                                    # checkerboard
    else:
        lab = (np.arange(N) % 3 == 0).astype(int)           # ...BAABAA...
    ordered = pe.BondCounter(2).add(*pe.bond_statistics(lab, nb, 2)).entropy().excess
    perm = pe.BondCounter(2)
    for _ in range(20):
        perm.add(*pe.bond_statistics(rng.permutation(lab), nb, 2))
    assert ordered < -1.0
    assert abs(perm.entropy().excess) < 0.01


def test_excess_never_positive():
    for _ in range(200):
        k = rng.integers(2, 6)
        Y = rng.random((k, k))
        Y = Y + Y.T
        Y /= Y.sum()
        assert pe.pair_entropy(Y.sum(1), Y, 12).excess <= 1e-12
        assert pe.mutual_information(Y) >= -1e-15


def test_input_validation():
    with pytest.raises(ValueError):
        pe.pair_entropy([0.5, 0.5], np.array([[0.5, 0.25], [0.25, 0.5]]), 12)   # does not sum to 1
    with pytest.raises(ValueError):
        pe.bethe_quasichemical(12, 0.25, -0.6)                                   # alpha not attainable at x = 0.25
