"""Command-line interface: ``pair-entropy chem`` (chemical term of one or more states) and ``pair-entropy excess``
(total excess entropy of alloy liquids against pure-liquid references)."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .excess import chemical_excess, excess_entropy
from .trajectory import read_trajectory


def _sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 23), b""):
            h.update(blk)
    return h.hexdigest()


def _read(path, a, pure=None):
    """Read one trajectory; ``pure`` names the species of every atom of a pure-liquid reference."""
    kw = {"skip": a.skip}
    if pure is not None:
        kw["types"] = [pure] * 64
    elif a.types:
        kw["types"] = a.types.split(",")
    if a.format != "ase":
        kw["units"] = a.lammps_units
    tr = read_trajectory(path, fmt=None if a.format == "auto" else a.format, **kw)
    if pure is not None and set(tr.symbols) != {pure}:
        raise SystemExit(f"{path}: reference given as {pure} but contains {tr.species()}")
    return tr


def _common(p):
    p.add_argument("--format", choices=("auto", "lammps", "ase"), default="auto",
                   help="trajectory format (default: LAMMPS for *.dump, *.lammpstrj, dump.*; ASE otherwise)")
    p.add_argument("--types", help="species of LAMMPS types 1, 2, ... (comma separated), when the dump has no "
                                   "element column")
    p.add_argument("--lammps-units", default="metal", choices=("metal", "real"), help="LAMMPS unit style")
    p.add_argument("--cutoff", choices=("partial", "total"), default="partial",
                   help="neighbour cutoff: first minima of the partial RDFs (default) or of the total RDF")
    p.add_argument("--rc", type=float, help="fixed neighbour cutoff, A (overrides --cutoff)")
    p.add_argument("--rc-delta", type=float, default=0.1, help="cutoff shift of the sensitivity cases, A")
    p.add_argument("--nblock", type=int, default=5, help="blocks for the standard errors")
    p.add_argument("--stride", type=int, default=1, help="frame stride for the neighbour graphs")
    p.add_argument("--rdf-stride", type=int, default=10, help="frame stride for the RDFs")
    p.add_argument("--skip", type=int, default=0, help="leading frames to discard")
    p.add_argument("--seed", type=int, default=0, help="seed of the label permutations")
    p.add_argument("-o", "--output", default=None, help="output JSON")


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    return o


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pair-entropy", description="Pair (quasi-chemical) chemical entropy and "
                                 "excess entropy of alloy liquids from MD trajectories (J/mol-atom/K).")
    ap.add_argument("--version", action="version", version=f"pair-entropy {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pc = sub.add_parser("chem", help="chemical term of each trajectory (positions only)")
    pc.add_argument("traj", nargs="+")
    pc.add_argument("--species", help="species order (comma separated); default: sorted species of each file")
    _common(pc)
    pe = sub.add_parser("excess", help="total excess entropy of alloys against pure-liquid references")
    pe.add_argument("--temp", type=float, required=True, help="temperature of every state, K")
    pe.add_argument("--dt", type=float, required=True, help="time between saved frames, fs")
    pe.add_argument("--ref", action="append", required=True,
                    help="pure-liquid trajectory, one per species; SPECIES=PATH names the species of a LAMMPS dump")
    pe.add_argument("--alloy", action="append", required=True, help="alloy trajectory (repeat for several)")
    pe.add_argument("--tmax", type=float, default=3.0, help="VACF window, ps")
    pe.add_argument("--ideal-tol", type=float, default=0.05, help="permutation-check tolerance, J/mol-atom/K")
    pe.add_argument("--order", type=int, default=None, help="highest Redlich-Kister order k (default: largest the "
                                                           "compositions determine, at most 2)")
    _common(pe)
    a = ap.parse_args(argv)
    settings = {k: v for k, v in vars(a).items() if k not in ("cmd",)}

    if a.cmd == "chem":
        res = {}
        for p in a.traj:
            tr = _read(p, a)
            sp = a.species.split(",") if a.species else tr.species()
            r = chemical_excess(tr, sp, a.cutoff, a.rc, a.rc_delta, a.stride, a.nblock, a.rdf_stride, a.seed)
            res[p] = r
            pcx = r["permutation_check"]["S_chem_excess_permuted"]
            comp = " ".join(f"{s}{x:.3f}" for s, x in r["x"].items())
            print(f"{p}: {comp}  zbar={r['zbar']:.2f}  S_chem,ex={r['S_chem_excess']:+.4f}"
                  f"({r['S_chem_excess_u']:.4f})  permuted={pcx:+.4f} J/mol-atom/K")
        out = {"tool": "pair-entropy chem", "version": __version__, "units": "J/mol-atom/K", "states": res}
        files = a.traj
    else:
        refs = {}
        ref_files = []
        for p in a.ref:
            el = None
            if "=" in p and p.split("=", 1)[0].isalpha():
                el, p = p.split("=", 1)
            tr = _read(p, a, pure=el)
            ref_files.append(p)
            sp = tr.species()
            if len(sp) != 1:
                ap.error(f"{p}: a reference must be a pure liquid, found {sp}")
            if sp[0] in refs:
                ap.error(f"two references for {sp[0]}")
            refs[sp[0]] = tr
        alloys = {p: _read(p, a) for p in a.alloy}
        out = excess_entropy(alloys, refs, a.temp, a.dt, a.cutoff, a.rc, a.rc_delta, a.tmax, a.nblock, a.stride,
                             a.rdf_stride, a.ideal_tol, a.seed, a.order)
        out = {"tool": "pair-entropy excess", "version": __version__, **out}
        files = ref_files + a.alloy
        for w in out["warnings"]:
            print("WARNING:", w, file=sys.stderr)
        for p, r in out["alloys"].items():
            comp = " ".join(f"{s}{x:.3f}" for s, x in r["x"].items() if x > 0)
            print(f"{p}: {comp}  S_vib,ex={r['S_vib_excess']:+.3f}({r['S_vib_excess_SE']:.3f})  "
                  f"S_chem,ex={r['S_chem_excess']:+.3f}({r['S_chem_excess_u']:.3f})  "
                  f"S_ex={r['S_excess']:+.3f}({r['S_excess_u']:.3f}) J/mol-atom/K  [{r['status']}]")
    out["settings"] = settings
    out["inputs_sha256"] = {p: _sha256(p) for p in files}
    out["citation"] = ("pair-entropy; pair estimator of the chemical-mixing entropy, replacing the chemical-mixing "
                       "term of Q.-J. Hong and Z.-K. Liu, Phys. Rev. Research 7, L012030 (2025)")
    if a.output:
        Path(a.output).write_text(json.dumps(_jsonable(out), indent=1))
        print(f"Written to {a.output}")
    return out


if __name__ == "__main__":  # pragma: no cover
    main()
