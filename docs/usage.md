# Usage

All entropies are in J/mol/K per mole of atoms (sites).

## Inputs

Trajectories are NVT production runs with a fixed orthorhombic cell.

- **LAMMPS text dumps** (`dump custom`): columns `id`, `type` or `element`, positions (`x y z`, `xu yu zu`, `xs ys zs` or `xsu ysu zsu`) and, for the vibrational term, `vx vy vz`. Give the species of the LAMMPS types with `types=["Cr", "Ni"]` (`--types Cr,Ni`) when there is no `element` column, and the unit style with `units="metal"` (velocities in Å/ps, default) or `"real"` (Å/fs). Masses are the standard atomic masses of the elements unless `masses={...}` is given. Files named `*.dump`, `*.lammpstrj` or `dump.*` are read as LAMMPS dumps.
- **ASE trajectories**: any file ASE reads with positions and, for the vibrational term, velocities (extxyz, `.traj`, ...).

```python
import pair_entropy as pe
tr = pe.read_lammps_dump("CrNi.dump", types=["Cr", "Ni"], units="metal", skip=100, stride=1)
tr = pe.read_ase("CrNi.traj")
```

Internally, positions are in Å, velocities in Å/fs and masses in amu (`pe.Trajectory`).

## The estimator on its own

```python
pe.pair_entropy(x, y, zbar)                 # x (k,), ordered bond fractions y (k, k), mean coordination
pe.bethe_quasichemical(z, x_B, alpha)       # binary with Warren-Cowley alpha on a lattice of coordination z
pe.mutual_information(y)                    # I, so that S_chem,ex = -R (zbar/2) I

N, nb = pe.lattice("fcc", 10)               # periodic neighbour table
bc = pe.BondCounter(k=2)
bc.add(*pe.bond_statistics(labels, nb, 2))  # pool counts over frames or samples
bc.entropy(), bc.bonds_per_pair(["A", "B"])
```

## Chemical term of a trajectory

```python
res = pe.chemical_excess(tr, ["Cr", "Ni"], cutoff="partial", rc=None, rc_delta=0.1,
                         stride=1, nblock=5, rdf_stride=10, seed=0)
```

- `cutoff="partial"` (default): two atoms are neighbours when closer than the first minimum of their partial radial distribution function; `"total"` uses the first minimum of the total one; `rc=` (scalar or k × k, Å) fixes it.
- Returned: `S_chem_excess`, its block standard error, the cutoff spread and their combination `S_chem_excess_u`; the k × k bond matrices, `bonds_per_pair`, `zbar`, the cutoffs used and every cutoff case; `S_chem_excess_x_reference_variant` (sensitivity case); and `permutation_check`, the excess of the same graphs with the labels permuted at random, which must be ~0.

## Excess entropy

```python
refs = {"Cr": pe.read_trajectory("Cr.dump", types=["Cr"]), "Ni": pe.read_trajectory("Ni.dump", types=["Ni"])}
alloys = {name: pe.read_trajectory(name, types=["Cr", "Ni"]) for name in ("Cr75Ni25.dump", "Cr50Ni50.dump")}
out = pe.excess_entropy(alloys, refs, T=2200, dt_fs=2.0, tmax_ps=3.0, nblock=5, ideal_tol=0.05)
```

Every state must be at the same temperature. For each alloy the result holds `S_vib_excess` and its standard error, the chemical term as above, `S_excess` and `S_excess_u`, the mean kinetic temperature and DOS diagnostics (`dos_integral_per_atom` should be 3, `negative_dos_weight_per_atom` small). The excess electronic entropy is taken as zero, an assumption recorded under `assumptions`. If any alloy fails the label-permutation check, all alloys are marked as held. Warnings flag VACF windows longer than a block, kinetic temperatures more than 5 % from `T`, and species pairs with fewer than 1000 sampled bonds under random mixing.

`out["redlich_kister"]` holds the compositions, `S_excess`, its covariance (including the shared pure-liquid references) and, when the compositions determine them, the generalised-least-squares b_k of S_ex = -Σ_{i<j} x_i x_j Σ_k b_k^{ij} (x_i - x_j)^k, species in alphabetical order. The highest order is the largest the compositions determine, at most 2 (`order=` to set it).

## Command line

    pair-entropy chem TRAJ [TRAJ ...] [--species Cr,Ni] [--types Cr,Ni] [--lammps-units metal]
                      [--cutoff partial|total] [--rc A] [--rc-delta 0.1] [--nblock 5] [--stride 1]
                      [--rdf-stride 10] [--skip 0] [--seed 0] [-o chem.json]

    pair-entropy excess --temp K --dt FS --ref [SPECIES=]PURE [--ref ...] --alloy TRAJ [--alloy TRAJ ...]
                        [--tmax 3] [--ideal-tol 0.05] [--order K] [options of chem] [-o excess.json]

For a pure-liquid LAMMPS dump without an `element` column, `--ref Ni=Ni.dump` names its species (`--types` applies to the alloys). The JSON output records the results, the settings and the SHA-256 hashes of the input files.

## Practical notes

- The neighbour-graph step builds an N × N distance matrix per frame; a few thousand atoms per frame are fine.
- Use enough frames that every species pair has well over 1000 sampled bonds, and blocks longer than the VACF window.
- The cutoff spread is part of the stated uncertainty; potential and method errors are not included.
