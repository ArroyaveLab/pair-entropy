"""Vibrational entropy from the mass-weighted velocity autocorrelation function (VACF).

The VACF of a state is windowed (Hann window of length t_max) and cosine-transformed to a vibrational density of
states g(nu), normalised to 3N, which is converted to entropy with the quantum harmonic entropy per mode.
"""
from __future__ import annotations

import numpy as np

from .estimator import R
from .graph import block_se

__all__ = ["vacf", "density_of_states", "harmonic_entropy", "vibrational_entropy", "kinetic_temperature"]

KB_EV = 8.617333262e-5           # eV/K
H_EV_S = 4.135667696e-15         # eV s
EV_PER_AMU_A2_FS2 = 103.642697   # 1 amu A^2 fs^-2 in eV
_trapz = getattr(np, "trapezoid", None) or np.trapz


def vacf(vel, masses):
    """Mass-weighted velocity autocorrelation sum_a m_a <v_a(0).v_a(t)>, amu A^2/fs^2. vel (nf, N, 3) in A/fs."""
    nf, N, _ = vel.shape
    nfft = 1 << int(np.ceil(np.log2(2 * nf)))
    masses = np.asarray(masses, float)
    C = np.zeros(nf)
    for a in range(0, N, 32):                                        # chunks of atoms bound the memory
        F = np.fft.rfft(vel[:, a:a + 32], n=nfft, axis=0)
        ac = np.fft.irfft((F * np.conj(F)).real, n=nfft, axis=0)[:nf]
        C += (ac.sum(2) * masses[None, a:a + 32]).sum(1)
    return C / (nf - np.arange(nf))


def density_of_states(vel, masses, dt_fs, T, tmax_ps=3.0, nnu=40001):
    """Vibrational DOS g(nu) with nu in THz, integral 3N. Returns (nu_THz, g)."""
    nf = vel.shape[0]
    C = vacf(vel, masses)
    t_fs = np.arange(nf) * dt_fs
    tmax = min(nf - 1, int(tmax_ps * 1000 / dt_fs))
    if tmax < 2:
        raise ValueError("the VACF window holds fewer than three frames")
    Cw = C[:tmax + 1] * 0.5 * (1 + np.cos(np.pi * np.arange(tmax + 1) / tmax))
    kT = KB_EV * T / EV_PER_AMU_A2_FS2
    nu = np.linspace(0, 1.0 / (2 * dt_fs) * 1000, nnu)
    tt = t_fs[:tmax + 1]
    g = np.empty_like(nu)
    for a in range(0, len(nu), 2000):
        n = nu[a:a + 2000]
        g[a:a + 2000] = 4.0 / kT * _trapz(Cw[None] * np.cos(2 * np.pi * n[:, None] * 1e-3 * tt[None]), tt, axis=1)
    return nu, g * 1e-3


def harmonic_entropy(nu_THz, g, T):
    """Quantum harmonic entropy of the DOS, J/mol/K for the whole cell (divide by N for per atom)."""
    x = H_EV_S * nu_THz * 1e12 / (2 * KB_EV * T)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        W = np.where(x > 1e-12, x / np.tanh(x) - np.log(2 * np.sinh(x)), 0.0)
    return R * _trapz(g * W, nu_THz)


def vibrational_entropy(traj, dt_fs, T, tmax_ps=3.0, nblock=5):
    """Vibrational entropy per atom (J/mol/K) of a trajectory with velocities, with block standard error."""
    if traj.velocities is None or traj.masses is None:
        raise ValueError(f"{traj.source}: velocities and masses are needed for the vibrational term")
    vel, m, N = traj.velocities, traj.masses, traj.natoms
    nu, g = density_of_states(vel, m, dt_fs, T, tmax_ps)
    full = harmonic_entropy(nu, g, T) / N
    nf = vel.shape[0]
    bl = []
    for b in range(nblock):
        sl = slice(b * nf // nblock, (b + 1) * nf // nblock)
        nub, gb = density_of_states(vel[sl], m, dt_fs, T, tmax_ps)
        bl.append(harmonic_entropy(nub, gb, T) / N)
    dnu = nu[1] - nu[0]
    return {"S_vib": float(full), "S_vib_blocks": [float(b) for b in bl], "S_vib_block_SE": block_se(bl),
            "dos_integral_per_atom": float(_trapz(g, nu) / N),
            "negative_dos_weight_per_atom": float(-g[g < 0].sum() * dnu / N)}


def kinetic_temperature(traj):
    """Mean kinetic temperature, K."""
    ke = 0.5 * (traj.masses[None, :, None] * traj.velocities ** 2).sum((1, 2)) * EV_PER_AMU_A2_FS2  # eV
    return float(ke.mean() * 2 / (3 * traj.natoms * KB_EV))
