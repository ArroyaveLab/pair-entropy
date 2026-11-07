import numpy as np

import pair_entropy as pe


def test_binary_recovers_bk():
    xs = np.array([0.25, 0.5, 0.75])
    b = np.array([-3.0, 1.2, 0.4])
    X = np.stack([1 - xs, xs], 1)
    sex = -xs * (1 - xs) * sum(b[k] * (1 - 2 * xs) ** k for k in range(3))
    fit = pe.fit_redlich_kister(X, sex, np.eye(3) * 0.01, 2)
    assert np.allclose(fit["b"], b, atol=1e-10)
    assert fit["dof"] == 0


def test_ternary_pairs_recovered():
    rng = np.random.default_rng(3)
    X = rng.dirichlet([1, 1, 1], size=12)
    b = np.array([-2.0, 0.5, -1.0, 0.3, 1.5, -0.2])        # (01: b0, b1), (02: b0, b1), (12: b0, b1)
    A, names = pe.excess.redlich_kister_design(X, 1)
    fit = pe.fit_redlich_kister(X, A @ b, np.eye(12), 1)
    assert names[:2] == [(0, 1, 0), (0, 1, 1)]
    assert np.allclose(fit["b"], b, atol=1e-9)
