# pair-entropy

Pair (quasi-chemical) estimator of the excess entropy of liquids from molecular dynamics.

The package computes the chemical-mixing entropy of an alloy liquid, with any number of species, as the pair (quasi-chemical) entropy of the species labels on the neighbour graph of the simulation,

    S_chem / R = -Σ_i x_i ln x_i - (z̄/2) Σ_ij y_ij ln[ y_ij / (w_i w_j) ],

where x_i is the site fraction of species i, y_ij the fraction of bonds with i at one end and j at the other, w_i = Σ_j y_ij the bond-end fraction and z̄ the mean coordination. This pair estimator replaces the chemical-mixing term of Hong and Liu, Phys. Rev. Research 7, L012030 (2025), https://doi.org/10.1103/PhysRevResearch.7.L012030. Since the second term is -(z̄/2) times the mutual information of the species at the two ends of a bond, random labels on any graph give the ideal entropy of mixing exactly, and any chemical order lowers the entropy. The expression is exact on a Bethe lattice and for a one-dimensional chain, and an approximation otherwise.

Together with the vibrational entropy from the velocity autocorrelation function, the package gives the excess entropy of the liquid,

    S_ex = S_vib,ex + S_chem,ex     (J/mol-atom/K; electronic term reported as missing),

with block standard errors, the spread over neighbour cutoffs, a label-permutation (ideal-mixing) check and, optionally, the Redlich–Kister b_k parameters of L_k = a_k + b_k T.

The derivation and its tests are in [docs/derivation.md](docs/derivation.md); usage is in [docs/usage.md](docs/usage.md).

## Install

    pip install git+https://github.com/ArroyaveLab/pair-entropy

or, from a clone, `pip install -e ".[test]"`. Requires Python ≥ 3.10, NumPy and ASE.

## Quick start

```python
import numpy as np
import pair_entropy as pe

# the estimator itself: site fractions, ordered bond fractions, mean coordination
x = [0.5, 0.5]
y = np.array([[0.20, 0.30], [0.30, 0.20]])
print(pe.pair_entropy(x, y, zbar=12))          # PairEntropy(total, ideal, excess), J/mol/K

# chemical term of an MD trajectory (LAMMPS dump or any ASE-readable file)
tr = pe.read_trajectory("CrNi.dump", types=["Cr", "Ni"], units="metal")
res = pe.chemical_excess(tr, ["Cr", "Ni"])     # partial-RDF first-minimum cutoffs by default
print(res["S_chem_excess"], res["S_chem_excess_u"], res["permutation_check"]["S_chem_excess_permuted"])
```

Command line:

    pair-entropy chem CrNi.dump --types Cr,Ni -o chem.json
    pair-entropy excess --temp 2200 --dt 2 --ref Cr=Cr.dump --ref Ni=Ni.dump \
        --alloy Cr75Ni25.dump --alloy Cr50Ni50.dump --alloy Cr25Ni75.dump --types Cr,Ni -o excess.json

## Tests

    python -m pytest

The tests check the ideal limit (machine precision on lattices and enumerated ensembles), the Bethe lattice against its exact Ising solution for Warren–Cowley α = ±0.1, ±0.3, ±0.6, the one-dimensional chain, the label-permutation check, the readers and the vibrational term on synthetic trajectories.

## Citation

If you use this package, please cite it (see [CITATION.cff](CITATION.cff)) together with Q.-J. Hong and Z.-K. Liu, Phys. Rev. Research 7, L012030 (2025).

## License

MIT, see [LICENSE](LICENSE).
