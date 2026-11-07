"""Trajectory container and readers (LAMMPS text dumps and ASE-readable files).

Internal units: positions in Angstrom, velocities in Angstrom/fs, masses in amu. Only orthorhombic, fixed cells are
supported (NVT production runs).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

__all__ = ["Trajectory", "read_ase", "read_lammps_dump", "read_trajectory"]

# velocity unit of each LAMMPS unit style, in Angstrom/fs (distance unit of those styles is Angstrom)
LAMMPS_VELOCITY_TO_A_PER_FS = {"metal": 1e-3, "real": 1.0}


@dataclass
class Trajectory:
    positions: np.ndarray              # (nframes, N, 3) Angstrom
    symbols: list                      # (N,) element symbols (or any species names)
    box: np.ndarray                    # (3,) orthorhombic edges, Angstrom
    velocities: np.ndarray | None = None   # (nframes, N, 3) Angstrom/fs
    masses: np.ndarray | None = None       # (N,) amu
    source: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.positions = np.asarray(self.positions, float)
        self.box = np.asarray(self.box, float).reshape(3)
        self.symbols = [str(s) for s in self.symbols]
        if self.positions.ndim != 3 or self.positions.shape[1:] != (len(self.symbols), 3):
            raise ValueError("positions must be (nframes, N, 3) with N = len(symbols)")
        if self.velocities is not None:
            self.velocities = np.asarray(self.velocities, float)
            if self.velocities.shape != self.positions.shape:
                raise ValueError("velocities must have the shape of positions")
        if self.masses is not None:
            self.masses = np.asarray(self.masses, float).reshape(len(self.symbols))

    @property
    def nframes(self) -> int:
        return self.positions.shape[0]

    @property
    def natoms(self) -> int:
        return len(self.symbols)

    def species(self) -> list:
        return sorted(set(self.symbols))

    def labels(self, species: Sequence[str]) -> np.ndarray:
        idx = {s: i for i, s in enumerate(species)}
        missing = set(self.symbols) - set(idx)
        if missing:
            raise ValueError(f"{self.source}: species {sorted(missing)} not in {list(species)}")
        return np.array([idx[s] for s in self.symbols], int)


def _default_masses(symbols):
    try:
        from ase.data import atomic_masses, atomic_numbers
    except ImportError:  # pragma: no cover
        return None
    try:
        return np.array([atomic_masses[atomic_numbers[s]] for s in symbols])
    except KeyError:
        return None


def read_ase(path, skip: int = 0, stride: int = 1, index=":") -> Trajectory:
    """Read any ASE-readable trajectory (extxyz, .traj, ...). Velocities are converted from ASE units to A/fs."""
    from ase import units
    from ase.io import iread

    pos, vel = [], []
    sym = cell = masses = None
    has_v = True
    for i, at in enumerate(iread(str(path), index=index)):
        if i < skip or (i - skip) % stride:
            continue
        if sym is None:
            sym, cell, masses = at.get_chemical_symbols(), at.cell.array.copy(), at.get_masses()
        elif at.get_chemical_symbols() != sym:
            raise ValueError(f"{path}: species order changes between frames")
        elif not np.allclose(at.cell.array, cell, atol=1e-6):
            raise ValueError(f"{path}: cell changes between frames (fixed-cell production expected)")
        pos.append(at.positions.copy())
        v = at.get_velocities()
        if v is None or not np.any(v):
            has_v = False
        else:
            vel.append(v * units.fs)
    if not pos:
        raise ValueError(f"{path}: no frames")
    if not np.allclose(cell, np.diag(np.diag(cell)), atol=1e-8):
        raise ValueError(f"{path}: only orthorhombic cells are supported")
    return Trajectory(np.array(pos), sym, np.diag(cell).copy(), np.array(vel) if has_v else None,
                      np.asarray(masses), source=str(path))


def read_lammps_dump(path, types: Sequence[str] | None = None, units: str = "metal", skip: int = 0,
                     stride: int = 1, masses: dict | None = None) -> Trajectory:
    """Read a LAMMPS text dump (``dump custom``) with an orthorhombic box.

    Columns: ``id``, ``type`` or ``element``, positions as x y z, xu yu zu, xs ys zs or xsu ysu zsu, and optionally
    vx vy vz. ``types`` maps LAMMPS types 1, 2, ... to species names (needed when there is no ``element`` column).
    ``units`` is the LAMMPS unit style ("metal": velocities in A/ps; "real": A/fs). ``masses`` optionally maps
    species to amu; the default is the standard atomic mass of each element (via ASE when installed).
    """
    if units not in LAMMPS_VELOCITY_TO_A_PER_FS:
        raise ValueError(f"units must be one of {sorted(LAMMPS_VELOCITY_TO_A_PER_FS)}")
    vconv = LAMMPS_VELOCITY_TO_A_PER_FS[units]
    pos, vel, sym, box0 = [], [], None, None
    has_v = None
    frame = -1
    with open(path) as fh:
        while True:
            line = fh.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            fh.readline()
            frame += 1
            if not fh.readline().startswith("ITEM: NUMBER OF ATOMS"):
                raise ValueError(f"{path}: malformed frame {frame}")
            n = int(fh.readline())
            hdr = fh.readline()
            if not hdr.startswith("ITEM: BOX BOUNDS"):
                raise ValueError(f"{path}: malformed frame {frame}")
            if "xy" in hdr.split():
                raise ValueError(f"{path}: triclinic boxes are not supported")
            bounds = np.array([[float(v) for v in fh.readline().split()[:2]] for _ in range(3)])
            box = bounds[:, 1] - bounds[:, 0]
            cols = fh.readline().split()[2:]
            rows = [fh.readline().split() for _ in range(n)]
            use = frame >= skip and (frame - skip) % stride == 0
            if not use:
                continue
            c = {name: i for i, name in enumerate(cols)}
            ids = np.array([int(r[c["id"]]) for r in rows]) if "id" in c else np.arange(n)
            order = np.argsort(ids)
            rows = [rows[i] for i in order]
            if "element" in c:
                s = [r[c["element"]] for r in rows]
            elif "type" in c:
                if types is None:
                    raise ValueError(f"{path}: give the species of each LAMMPS type (types=...)")
                s = [types[int(r[c["type"]]) - 1] for r in rows]
            else:
                raise ValueError(f"{path}: dump needs a 'type' or 'element' column")
            for trip, scaled in ((("x", "y", "z"), False), (("xu", "yu", "zu"), False),
                                 (("xs", "ys", "zs"), True), (("xsu", "ysu", "zsu"), True)):
                if all(t in c for t in trip):
                    p = np.array([[float(r[c[t]]) for t in trip] for r in rows])
                    p = p * box if scaled else p - bounds[:, 0]
                    break
            else:
                raise ValueError(f"{path}: no position columns")
            hv = all(t in c for t in ("vx", "vy", "vz"))
            if has_v is None:
                has_v = hv
            if sym is None:
                sym, box0 = s, box
            elif s != sym:
                raise ValueError(f"{path}: species order changes between frames")
            elif not np.allclose(box, box0, atol=1e-6):
                raise ValueError(f"{path}: box changes between frames (fixed-cell production expected)")
            pos.append(p)
            if has_v:
                vel.append(np.array([[float(r[c[t]]) for t in ("vx", "vy", "vz")] for r in rows]) * vconv)
    if not pos:
        raise ValueError(f"{path}: no frames")
    if masses is not None:
        m = np.array([masses[s] for s in sym], float)
    else:
        m = _default_masses(sym)
    return Trajectory(np.array(pos), sym, box0, np.array(vel) if has_v else None, m, source=str(path),
                      meta={"lammps_units": units})


def read_trajectory(path, fmt: str | None = None, **kw) -> Trajectory:
    """Read a trajectory; ``fmt`` "lammps" for a LAMMPS text dump, "ase" for anything ASE reads. By default files
    named ``*.dump``, ``*.lammpstrj`` or ``dump.*`` are read as LAMMPS dumps and everything else with ASE."""
    p = Path(path)
    if fmt is None:
        fmt = "lammps" if (p.suffix in (".dump", ".lammpstrj") or p.name.startswith("dump.")) else "ase"
    if fmt == "lammps":
        return read_lammps_dump(path, **kw)
    if fmt == "ase":
        kw = {k: v for k, v in kw.items() if k in ("skip", "stride")}
        return read_ase(path, **kw)
    raise ValueError(f"unknown format {fmt!r}")
