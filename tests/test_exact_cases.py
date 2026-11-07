"""Ordered cases with exact entropies obtained by routes independent of the pair formula: the Bethe lattice (Ising
partition function, closed form at x = 0.5 and thermodynamic integration of the exact cavity solution at x != 0.5),
the one-dimensional Markov chain (entropy rate) and, as a loopy-lattice bound, the square-lattice Ising model
(Onsager)."""
import math

import numpy as np
import pytest

import pair_entropy as pe
from pair_entropy import R, shannon

ALPHAS = [-0.6, -0.3, -0.1, 0.1, 0.3, 0.6]
rng = np.random.default_rng(2)
_trapz = getattr(np, "trapezoid", None) or np.trapz


# ---------------------------------------------------------------------------------------------- Bethe lattice
def bethe_ising_entropy_half(z, alpha):
    """Exact entropy per site (J/mol/K) of the zero-field Ising model on a Bethe lattice with nearest-neighbour
    correlation alpha = tanh K (x_B = 0.5): from ln Z per site, ln 2 + (z/2) ln cosh K, and the energy."""
    K = math.atanh(alpha)
    return R * (math.log(2) + z / 2 * (math.log(math.cosh(K)) - K * math.tanh(K)))


def _pair_probs(K, hc):
    s = np.array([1.0, -1.0])
    P = np.exp(K * np.outer(s, s) + hc * (s[:, None] + s[None, :]))
    return P / P.sum()


def bethe_ising_entropy_integrated(z, x_B, alpha, n=20001):
    """Exact entropy per site (J/mol/K) of the quasi-chemical measure (site fraction x_B, Warren-Cowley alpha) on a
    Bethe lattice, by thermodynamic integration of the exact Ising solution, without the pair formula.

    Spin +1 = A, -1 = B. On a Bethe lattice the uniform state at coupling K and field h has the cavity field hc with
    hc = h + (z - 1) atanh(tanh K tanh hc), bond distribution P(s, s') ~ exp(K s s' + hc (s + s')) and site
    magnetisation m = tanh(h + z atanh(tanh K tanh hc)). The free energy obeys d(-beta f) = (z/2) <s s'> dK + m dh,
    with -beta f = ln(2 cosh h) at K = 0. It is integrated along K' = lambda K at the fixed cavity field of the target
    state (h follows from the cavity equation, so every point is an exact solution on the target branch), and
    s = -beta f - K (z/2) <s s'> - h m. Returns (entropy, m at the target)."""
    xA = 1 - x_B
    yAB = xA * x_B * (1 - alpha)
    PAA, PBB = xA - yAB, x_B - yAB
    K = 0.25 * math.log(PAA * PBB / yAB ** 2)
    hc = 0.25 * math.log(PAA / PBB)
    lam = np.linspace(0.0, 1.0, n)
    Ks = lam * K
    B = np.arctanh(np.tanh(Ks) * math.tanh(hc))
    hs = hc - (z - 1) * B
    ms = np.tanh(hs + z * B)
    corr = np.array([(lambda P: P[0, 0] + P[1, 1] - 2 * P[0, 1])(_pair_probs(k, hc)) for k in Ks])
    phi = math.log(2 * math.cosh(hs[0])) + _trapz((z / 2) * corr * np.gradient(Ks, lam) + ms * np.gradient(hs, lam),
                                                  lam)
    s = phi - K * (z / 2) * corr[-1] - hs[-1] * ms[-1]
    return R * s, ms[-1]


@pytest.mark.parametrize("z", [3, 4, 8, 12])
@pytest.mark.parametrize("alpha", ALPHAS)
def test_bethe_lattice_exact_half(z, alpha):
    """x_B = 0.5, alpha in +-0.1, +-0.3, +-0.6: the exact Bethe-lattice entropy equals the analytic quasi-chemical
    (pair) value to machine precision, and the excess is negative for ordering and for clustering."""
    exact = bethe_ising_entropy_half(z, alpha)
    s = pe.bethe_quasichemical(z, 0.5, alpha)
    assert s.total == pytest.approx(exact, abs=1e-12)
    assert s.excess < 0


@pytest.mark.parametrize("z", [4, 12])
@pytest.mark.parametrize("x_B,alpha", [(0.5, a) for a in ALPHAS] + [(0.25, a) for a in (-0.3, -0.1, 0.1, 0.3, 0.6)]
                         + [(0.1, -0.1), (0.1, 0.3)])
def test_bethe_lattice_exact_thermodynamic_integration(z, x_B, alpha):
    """Any x_B: thermodynamic integration of the exact Bethe-lattice solution (no use of the pair formula) gives the
    analytic quasi-chemical value."""
    exact, m = bethe_ising_entropy_integrated(z, x_B, alpha)
    assert (1 - m) / 2 == pytest.approx(x_B, abs=1e-10)      # the cavity solution has the target composition
    assert pe.bethe_quasichemical(z, x_B, alpha).total == pytest.approx(exact, abs=1e-6)


# ---------------------------------------------------------------------------------------------- 1-D chain
@pytest.mark.parametrize("x,alpha", [(0.3, -0.3), (0.5, 0.5), (0.1, 0.4), (0.25, -0.2), (0.5, -0.6)])
def test_markov_chain_exact(x, alpha):
    """Stationary binary Markov chain with Warren-Cowley alpha: the exact entropy rate H(next | current) equals the
    pair estimator with z = 2; a sampled ring of 2 x 10^5 sites agrees within sampling error."""
    xA = 1 - x
    P = np.array([[1 - x * (1 - alpha), x * (1 - alpha)], [xA * (1 - alpha), 1 - xA * (1 - alpha)]])
    rate = R * (xA * shannon(P[0]) + x * shannon(P[1]))
    assert pe.bethe_quasichemical(2, x, alpha).total == pytest.approx(rate, abs=1e-12)
    N = 200_000
    u = rng.random(N)
    lab = np.empty(N, int)
    lab[0] = 0
    for i in range(1, N):
        lab[i] = u[i] < P[lab[i - 1], 1]
    bc = pe.BondCounter(2).add(*pe.bond_statistics(lab, pe.lattice("chain", N)[1], 2))
    assert bc.entropy().total == pytest.approx(rate, abs=0.02)


# ---------------------------------------------------------------------------------------------- square Ising
def _ellipk(m):
    """Complete elliptic integral of the first kind K(m), by the arithmetic-geometric mean."""
    a, b = 1.0, math.sqrt(1 - m)
    while abs(a - b) > 1e-15:
        a, b = (a + b) / 2, math.sqrt(a * b)
    return math.pi / (2 * a)


def onsager(K):
    """Exact square-lattice Ising, zero field, K >= 0: entropy per site (nats) and nearest-neighbour correlation."""
    k = 2 * math.sinh(2 * K) / math.cosh(2 * K) ** 2
    t = np.linspace(0, math.pi, 20001)
    integ = _trapz(np.log((1 + np.sqrt(1 - k * k * np.sin(t) ** 2)) / 2), t)
    mbf = math.log(2 * math.cosh(2 * K)) + integ / (2 * math.pi)
    u = -(1 / math.tanh(2 * K)) * (1 + (2 / math.pi) * (2 * math.tanh(2 * K) ** 2 - 1) * _ellipk(k * k))
    return mbf + K * u, -u / 2


def test_square_ising_pair_approximation_bounds():
    """Loopy lattice (Onsager): the pair estimator at the exact correlation has the sign of the exact excess and
    overestimates its magnitude by less than 25 % for K <= 0.38 (it is an approximation off trees)."""
    assert onsager(1e-6)[0] == pytest.approx(math.log(2), abs=1e-6)
    for K in (0.2, 0.3, 0.38):
        s, c = onsager(K)
        ex_exact = R * s - pe.ideal_mixing(0.5)
        y = np.array([[(1 + c) / 4, (1 - c) / 4], [(1 - c) / 4, (1 + c) / 4]])
        ex_pair = pe.pair_entropy([0.5, 0.5], y, 4).excess
        assert ex_exact < 0 and ex_pair < 0
        assert 1.0 <= ex_pair / ex_exact < 1.25
