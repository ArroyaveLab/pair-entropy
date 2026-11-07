# The pair estimator of the chemical-mixing entropy

## Setting

In the method of Hong and Liu [1], the entropy of a liquid simulated by molecular dynamics is separated into a vibrational part, obtained from the velocity autocorrelation function, and a configurational part that accounts for the arrangement of the chemical species. The configurational part can be expressed as the entropy of the species labels on the neighbour graph of the simulation, in which two atoms are connected when their distance is smaller than the first minimum of the corresponding partial radial distribution function. Whatever the estimator, this entropy has a known limit: when the labels are distributed at random over the sites of the graph, it must reduce to the ideal entropy of mixing, -R Σ_i x_i ln x_i, independently of the structure of the graph. The package replaces the chemical-mixing term of Hong and Liu [1] with the pair approximation described below, which satisfies this limit exactly.

## The estimator

In the pair (quasi-chemical) approximation [2, 3], the entropy per site of a lattice with coordination z is obtained from the entropy of the bonds minus the entropy of the sites that each bond counts twice,

    S / (N k_B) = (z/2) H(y) - (z - 1) H(x),

where H denotes the Shannon entropy, x_i the fraction of species i and y_ij the fraction of bonds with i at one end and j at the other. For the neighbour graph of a liquid, in which the coordination varies from atom to atom, the same expression is written in terms of the mean coordination z̄ and of the bond-end fractions w_i = Σ_j y_ij, which reduce to x_i on a lattice:

    S_chem / R = -Σ_i x_i ln x_i - (z̄/2) Σ_ij y_ij ln[ y_ij / (w_i w_j) ].

The second term is -(z̄/2) times the mutual information I of the species at the two ends of a bond,

    I = Σ_ij y_ij ln[ y_ij / (w_i w_j) ] ≥ 0,

so that the excess chemical entropy is

    S_chem,ex = S_chem - S_ideal = -R (z̄/2) I.

On a lattice, w_i = x_i and Σ_j y_ij = x_i, and the identity H(y) = 2 H(x) - I turns the lattice form above into the expression for S_chem with z̄ = z.

## Properties

This form has three properties that make it suitable as an estimator.

1. **Ideal limit.** For random labels on any graph, y_ij = w_i w_j, so that I = 0 and the entropy reduces exactly to the ideal entropy of mixing. Ideal mixing therefore enters exactly once, whatever the coordination distribution of the graph.
2. **Sign.** Since I ≥ 0, any chemical order, whether ordering (Warren–Cowley α < 0) or clustering (α > 0), lowers the entropy below the ideal value.
3. **Exact cases.** The expression is exact for the quasi-chemical distribution on a Bethe lattice and for a one-dimensional Markov chain (z = 2). In other cases, for instance on lattices with loops, it remains an approximation.

For a binary A–B with site fraction x_B and Warren–Cowley parameter α on a lattice, the bond fractions are y_AB = y_BA = x_A x_B (1 - α), y_AA = x_A - y_AB and y_BB = x_B - y_AB (`pair_entropy.bethe_quasichemical`).

### Why the bond-end fractions

In a liquid, the species of an atom and its number of neighbours are correlated (a large atom has more neighbours than a small one). If the bond statistics were referred to the site fractions x_i, rather than to the bond-end fractions w_i, this coupling between species and coordination would be counted as chemical order, and random labels would not give the ideal entropy of mixing. The x_i-referenced form is still reported (`S_chem_excess_x_reference_variant`) as a sensitivity case.

## Tests of the formalism

The test suite checks each of these statements against results obtained independently of the pair formula.

| Case | Exact result | Test |
|---|---|---|
| Random labels, thermodynamic limit, z = 2–12, two to four species | -R Σ x ln x | `test_ideal_limit.py` (machine precision) |
| Random labels, every labelling of a 12-site ring and a 4 × 4 square torus enumerated | -R Σ x ln x | `test_ideal_limit.py` (machine precision) |
| Random labels sampled on chain, square, bcc and fcc lattices and on irregular (liquid-like) graphs | -R Σ x ln x | `test_ideal_limit.py` (within 0.01 J/mol/K) |
| Ordered labels permuted at random on the same graph | S_chem,ex ≈ 0 | `test_ideal_limit.py`, `test_trajectories.py` |
| Bethe lattice, x_B = 0.5, α = ±0.1, ±0.3, ±0.6, z = 3–12 | Ising partition function, ln 2 + (z/2)(ln cosh K - K tanh K), tanh K = α | `test_exact_cases.py` (machine precision) |
| Bethe lattice, x_B = 0.1–0.5 | thermodynamic integration of the exact cavity solution | `test_exact_cases.py` (within 10⁻⁶ J/mol/K) |
| One-dimensional Markov chain | entropy rate | `test_exact_cases.py` |
| Square-lattice Ising, x_B = 0.5 (loops) | Onsager; pair value of the same sign, magnitude within 25 % for K ≤ 0.38 | `test_exact_cases.py` |
| Harmonic oscillators on fcc sites with imposed order (ASE and LAMMPS input) | lattice pair value; harmonic vibrational entropy | `test_trajectories.py` |

## From molecular dynamics to the excess entropy

**Neighbour graph.** For every frame, two atoms of species i and j are neighbours when they are closer than r_c,ij, the first minimum (after the first peak) of the partial radial distribution function g_ij(r), computed from the same trajectory (default). The first minimum of the total radial distribution function, or a fixed cutoff, can be used instead. Bond counts y_ij, site counts and coordinations are pooled over the frames.

**Label-permutation check.** Before an excess entropy is accepted, the species labels of each alloy are permuted at random on every neighbour graph of the simulation, which is kept as built with the true labels. The pair estimator must then give |S_chem,ex| below a tolerance (default 0.05 J/mol/K); otherwise the results are marked as held.

**Uncertainty of the chemical term.** The block standard error over `nblock` consecutive blocks of frames is combined in quadrature with the cutoff spread, the largest change of S_chem,ex over the cutoff cases (base, base ∓ 0.1 Å, and the other radial-distribution-function convention).

**Vibrational term.** The mass-weighted velocity autocorrelation function of each state is multiplied by a Hann window of length t_max (default 3 ps) and cosine-transformed to a vibrational density of states g(ν), normalised to 3N, which is converted with the quantum harmonic entropy per mode,

    S_vib = R ∫ g(ν) [ u coth u - ln(2 sinh u) ] dν,   u = hν / (2 k_B T).

S_vib,ex is the alloy value minus the composition-weighted values of the pure liquids at the same temperature, with block standard errors.

**Excess entropy.** Per mole of atoms,

    S_ex = S_vib,ex + S_chem,ex.

The electronic term is not computed and is reported as missing, not as zero. For use in a CALPHAD assessment, the excess entropies of several compositions can be fitted, with the covariance introduced by the shared pure-liquid references, to the temperature derivative of a Redlich–Kister excess with L_k = a_k + b_k T,

    S_ex = -Σ_{i<j} x_i x_j Σ_k b_k^{ij} (x_i - x_j)^k,

which for a binary with x = x_B reads S_ex = -x (1 - x) Σ_k b_k (1 - 2x)^k.

## References

1. Q.-J. Hong and Z.-K. Liu, Phys. Rev. Research 7, L012030 (2025), https://doi.org/10.1103/PhysRevResearch.7.L012030.
2. E. A. Guggenheim, Proc. R. Soc. Lond. A 183, 203 (1944), https://doi.org/10.1098/rspa.1944.0032.
3. R. Kikuchi, Phys. Rev. 81, 988 (1951), https://doi.org/10.1103/PhysRev.81.988.
