"""End to end on synthetic trajectories: atoms vibrating harmonically on fcc sites with imposed chemical order, read
from ASE files and from LAMMPS dumps. The chemical term must equal the lattice pair value, the vibrational term the
harmonic value, and the label-permutation check must give ~0."""
import json
import math
import subprocess
import sys

import numpy as np
import pytest

import pair_entropy as pe
from pair_entropy import R
from pair_entropy.vibrational import EV_PER_AMU_A2_FS2, H_EV_S, KB_EV

A0, NCELL, T, DT, NF = 3.6, 3, 1000.0, 5.0, 1000
MASS = {"Cu": 63.546, "Ni": 58.6934, "Zr": 91.224}
RC_GAP = 0.5 * (A0 / math.sqrt(2) + A0)        # between the first (a/sqrt 2) and second (a) fcc shells


def _fcc():
    M = 2 * NCELL
    g = np.indices((M, M, M)).reshape(3, -1).T
    pts = g[g.sum(1) % 2 == 0]
    sub = {(0, 0, 0): 0, (1, 1, 0): 1, (1, 0, 1): 2, (0, 1, 1): 3}
    return pts * A0 / 2, np.array([sub[tuple(p % 2)] for p in pts]), NCELL * A0


def _frames(symbols, nu_THz, seed):
    """Each Cartesian component a classical harmonic oscillator at its frequency with equipartition amplitude at T.
    Returns positions (A), velocities (A/fs), box edge."""
    r = np.random.default_rng(seed)
    x0, _, Lb = _fcc()
    N = len(x0)
    m = np.array([MASS[s] for s in symbols])
    w = 2 * np.pi * np.asarray(nu_THz, float)[:, None] * 1e-3          # rad/fs
    kT = KB_EV * T / EV_PER_AMU_A2_FS2
    amp = np.sqrt(2 * kT / m)[:, None] * np.ones((N, 3))               # A/fs
    ph = r.uniform(0, 2 * np.pi, (N, 3))
    t = np.arange(NF)[:, None, None] * DT
    pos = x0[None] + amp / w * np.sin(w * t + ph)
    vel = amp * np.cos(w * t + ph)
    return pos, vel, Lb


def write_ase(path, symbols, nu, seed):
    from ase import Atoms, units
    from ase.io import write
    pos, vel, Lb = _frames(symbols, nu, seed)
    frames = []
    for f in range(NF):
        at = Atoms(symbols, positions=pos[f], cell=[Lb] * 3, pbc=True)
        at.set_velocities(vel[f] / units.fs)
        frames.append(at)
    write(str(path), frames)
    return str(path)


def write_lammps(path, symbols, nu, seed, element_column=False):
    """LAMMPS text dump, metal units (velocities in A/ps), atoms written in shuffled id order, wrapped positions."""
    pos, vel, Lb = _frames(symbols, nu, seed)
    types = sorted(set(symbols))
    N = len(symbols)
    order = np.random.default_rng(seed + 100).permutation(N)
    with open(path, "w") as fh:
        for f in range(NF):
            fh.write(f"ITEM: TIMESTEP\n{f * 5}\nITEM: NUMBER OF ATOMS\n{N}\nITEM: BOX BOUNDS pp pp pp\n")
            for _ in range(3):
                fh.write(f"-1.0 {Lb - 1.0}\n")
            col = "element" if element_column else "type"
            fh.write(f"ITEM: ATOMS id {col} x y z vx vy vz\n")
            p = (pos[f] % Lb) - 1.0
            for a in order:
                tag = symbols[a] if element_column else types.index(symbols[a]) + 1
                fh.write(f"{a + 1} {tag} {p[a, 0]:.8f} {p[a, 1]:.8f} {p[a, 2]:.8f} "
                         f"{vel[f, a, 0] * 1e3:.8f} {vel[f, a, 1] * 1e3:.8f} {vel[f, a, 2] * 1e3:.8f}\n")
    return str(path)


def s_harm(nu):
    x = H_EV_S * nu * 1e12 / (2 * KB_EV * T)
    return 3 * R * (x / math.tanh(x) - math.log(2 * math.sinh(x)))


def lattice_pair_excess(lab, k):
    """Pair excess on the ideal fcc nearest-neighbour table (independent of the RDF and cutoff route)."""
    x0, _, Lb = _fcc()
    d = x0[:, None] - x0[None]
    d -= Lb * np.round(d / Lb)
    r = np.sqrt((d ** 2).sum(-1))
    nb = np.array([np.where(np.abs(row - A0 / math.sqrt(2)) < 1e-6)[0] for row in r])
    assert nb.shape[1] == 12
    return pe.BondCounter(k).add(*pe.bond_statistics(lab, nb, k)).entropy().excess


@pytest.fixture(scope="module")
def cuni(tmp_path_factory):
    """Cu-Ni with L1_0 order at x = 0.5, and the pure references; ASE and LAMMPS copies."""
    d = tmp_path_factory.mktemp("cuni")
    x0, sub, _ = _fcc()
    N = len(x0)
    sym = list(np.where(sub < 2, "Cu", "Ni"))
    nu = {"Cu": 8.0, "Ni": 10.0, "CuNi": 9.5}
    f = {"Cu": write_ase(d / "Cu.traj", ["Cu"] * N, [nu["Cu"]] * N, 1),
         "Ni": write_ase(d / "Ni.extxyz", ["Ni"] * N, [nu["Ni"]] * N, 2),
         "CuNi": write_ase(d / "CuNi.traj", sym, [nu["CuNi"]] * N, 3),
         "Cu_dump": write_lammps(d / "Cu.dump", ["Cu"] * N, [nu["Cu"]] * N, 1),
         "Ni_dump": write_lammps(d / "Ni.dump", ["Ni"] * N, [nu["Ni"]] * N, 2),
         "CuNi_dump": write_lammps(d / "CuNi.dump", sym, [nu["CuNi"]] * N, 3),
         "CuNi_dump_el": write_lammps(d / "dump.CuNi_el", sym, [nu["CuNi"]] * N, 3, element_column=True)}
    return d, f, sym, nu


def test_lammps_reader_matches_ase(cuni):
    _, f, sym, _ = cuni
    a = pe.read_ase(f["CuNi"])
    b = pe.read_lammps_dump(f["CuNi_dump"], types=["Cu", "Ni"], units="metal")
    c = pe.read_trajectory(f["CuNi_dump_el"])
    for t in (b, c):
        assert t.symbols == a.symbols == sym
        assert np.allclose(t.box, a.box)
        dpos = t.positions - a.positions
        dpos -= a.box * np.round(dpos / a.box)                     # wrapping and origin shift only
        assert np.abs(dpos).max() < 1e-6
        assert np.allclose(t.velocities, a.velocities, atol=1e-9)
        assert np.allclose(t.masses, a.masses, atol=1e-3)
    s = pe.read_lammps_dump(f["CuNi_dump"], types=["Cu", "Ni"], skip=10, stride=3)
    assert s.nframes == len(range(10, NF, 3))


def test_partial_rdf_cutoffs_in_shell_gap(cuni):
    _, f, _, _ = cuni
    cut = pe.rdf_cutoffs(pe.read_ase(f["CuNi"]), ["Cu", "Ni"])
    assert A0 / math.sqrt(2) < cut["rc_total_A"] < A0
    assert np.all((cut["rc_partial_A"] > A0 / math.sqrt(2)) & (cut["rc_partial_A"] < A0))


@pytest.mark.parametrize("which", ["CuNi", "CuNi_dump"])
def test_chemical_term_binary_ordered(cuni, which):
    """L1_0 at x = 0.5 on fcc, default partial-RDF cutoffs: S_chem,ex = -6 R I exactly (the lattice value), zero
    cutoff spread, and the label-permutation check gives ~0."""
    _, f, sym, _ = cuni
    tr = pe.read_trajectory(f[which], types=["Cu", "Ni"])
    r = pe.chemical_excess(tr, ["Cu", "Ni"], stride=5, rdf_stride=20)
    y = np.array([[1 / 6, 1 / 3], [1 / 3, 1 / 6]])
    assert r["S_chem_excess"] == pytest.approx(-R * 6 * (y * np.log(y / 0.25)).sum(), abs=1e-9)
    assert r["S_chem_excess"] == pytest.approx(lattice_pair_excess((np.array(sym) == "Ni").astype(int), 2), abs=1e-9)
    assert r["zbar"] == pytest.approx(12.0)
    assert 0 <= r["S_chem_excess_cutoff_spread"] < 0.05     # base -/+ 0.1 A and the total-RDF minimum
    assert r["cutoff_case"] == "partial_rdf_minimum"
    assert r["bonds_per_pair"]["Cu-Ni"] == pytest.approx(r["frames_in_graph"] * len(sym) * 8 / 2)
    assert abs(r["permutation_check"]["S_chem_excess_permuted"]) < 0.01


def test_excess_entropy_binary(cuni):
    """Full excess entropy: S_vib from the VACF matches the harmonic value, S_ex = S_vib,ex + S_chem,ex,
    b_0 = -4 S_ex at x = 0.5, the electronic term is reported as missing, and an impossible permutation-check
    tolerance holds the result."""
    _, f, _, nu = cuni
    refs = {"Cu": pe.read_ase(f["Cu"]), "Ni": pe.read_ase(f["Ni"])}
    alloys = {"CuNi": pe.read_ase(f["CuNi"])}
    out = pe.excess_entropy(alloys, refs, T, DT, rc=RC_GAP, tmax_ps=0.5, stride=5)
    a = out["alloys"]["CuNi"]
    assert out["permutation_check"]["passed"] and a["status"] == "computed"
    for el in ("Cu", "Ni"):
        assert out["references"][el]["S_vib"] == pytest.approx(s_harm(nu[el]), rel=0.01)
        assert out["references"][el]["dos_integral_per_atom"] == pytest.approx(3.0, rel=0.01)
    expect = s_harm(nu["CuNi"]) - 0.5 * s_harm(nu["Cu"]) - 0.5 * s_harm(nu["Ni"])
    assert a["S_vib_excess"] == pytest.approx(expect, abs=0.05)
    assert a["S_excess"] == pytest.approx(a["S_vib_excess"] + a["S_chem_excess"], abs=1e-12)
    assert np.isfinite(a["S_excess_u"]) and a["S_excess_u"] > 0
    assert a["T_kinetic_K"] == pytest.approx(T, rel=0.05)
    fit = out["redlich_kister"]["fit"]
    assert fit["order"] == 0 and fit["b"][0] == pytest.approx(-a["S_excess"] / 0.25, rel=1e-9)
    assert out["missing_terms"]["electronic"].startswith("not computed")
    held = pe.excess_entropy(alloys, refs, T, DT, rc=RC_GAP, tmax_ps=0.5, stride=5, ideal_tol=1e-12)
    assert not held["permutation_check"]["passed"]
    assert held["alloys"]["CuNi"]["status"].startswith("held")


def test_ternary_ordered(tmp_path):
    """Cu-Ni-Zr on fcc (Cu on two sublattices, Ni and Zr on one each): a 3 x 3 bond matrix, no Ni-Ni or Zr-Zr
    bonds, the lattice pair value, and the permutation check ~0."""
    x0, sub, _ = _fcc()
    N = len(x0)
    sym = list(np.array(["Cu", "Cu", "Ni", "Zr"])[sub])
    tr = pe.read_ase(write_ase(tmp_path / "CuNiZr.traj", sym, [8.0] * N, 7))
    r = pe.chemical_excess(tr, ["Cu", "Ni", "Zr"], rc=RC_GAP, stride=10)
    assert r["x"] == pytest.approx({"Cu": 0.5, "Ni": 0.25, "Zr": 0.25})
    lab = np.array([{"Cu": 0, "Ni": 1, "Zr": 2}[s] for s in sym])
    assert r["S_chem_excess"] == pytest.approx(lattice_pair_excess(lab, 3), abs=1e-9)
    assert r["S_chem_excess"] < -0.5
    assert np.array(r["y_bonds"]).shape == (3, 3)
    assert r["bonds_per_pair"]["Ni-Ni"] == 0 and r["bonds_per_pair"]["Zr-Zr"] == 0
    assert sum(r["bonds_per_pair"].values()) == pytest.approx(r["frames_in_graph"] * N * 12 / 2)
    assert abs(r["permutation_check"]["S_chem_excess_permuted"]) < 0.02


def test_cli(cuni, tmp_path):
    _, f, _, _ = cuni
    out = tmp_path / "chem.json"
    cp = subprocess.run([sys.executable, "-m", "pair_entropy.cli", "chem", f["CuNi_dump"], "--types", "Cu,Ni",
                         "--stride", "10", "--rdf-stride", "50", "-o", str(out)], capture_output=True, text=True)
    assert cp.returncode == 0, cp.stderr
    r = json.loads(out.read_text())["states"][f["CuNi_dump"]]
    y = np.array([[1 / 6, 1 / 3], [1 / 3, 1 / 6]])
    assert r["S_chem_excess"] == pytest.approx(-R * 6 * (y * np.log(y / 0.25)).sum(), abs=1e-9)
    out2 = tmp_path / "excess.json"
    cp = subprocess.run([sys.executable, "-m", "pair_entropy.cli", "excess", "--temp", str(T), "--dt", str(DT),
                         "--ref", f["Cu"], "--ref", "Ni=" + f["Ni_dump"], "--alloy", f["CuNi_dump"], "--types", "Cu,Ni",
                         "--rc", str(RC_GAP), "--tmax", "0.5", "--stride", "10", "-o", str(out2)],
                        capture_output=True, text=True)
    assert cp.returncode == 0, cp.stderr
    o = json.loads(out2.read_text())
    assert o["permutation_check"]["passed"]
    assert set(o["inputs_sha256"]) == {f["Cu"], f["Ni_dump"], f["CuNi_dump"]}
    assert set(o["references"]) == {"Cu", "Ni"}
    # same answer from the API on the ASE copies (LAMMPS velocities and masses read consistently)
    api = pe.excess_entropy({"a": pe.read_ase(f["CuNi"])}, {"Cu": pe.read_ase(f["Cu"]), "Ni": pe.read_ase(f["Ni"])},
                            T, DT, rc=RC_GAP, tmax_ps=0.5, stride=10)
    assert o["alloys"][f["CuNi_dump"]]["S_excess"] == pytest.approx(api["alloys"]["a"]["S_excess"], abs=1e-3)
