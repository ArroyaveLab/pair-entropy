"""pair-entropy: pair (quasi-chemical) estimator of the chemical-mixing entropy of alloy liquids from molecular
dynamics, with the vibrational term from the velocity autocorrelation function and the total excess entropy."""

__version__ = "0.1.0"

from .estimator import (R, BondCounter, PairEntropy, adjacency_statistics, bethe_quasichemical,  # noqa: E402
                        bond_statistics, ideal_mixing, mutual_information, pair_entropy, pair_entropy_x_reference,
                        shannon)
from .excess import chemical_excess, excess_entropy, fit_redlich_kister  # noqa: E402
from .graph import first_minimum, neighbour_graph, partial_rdfs, permutation_check, rdf_cutoffs  # noqa: E402
from .lattices import lattice  # noqa: E402
from .trajectory import Trajectory, read_ase, read_lammps_dump, read_trajectory  # noqa: E402
from .vibrational import density_of_states, harmonic_entropy, vibrational_entropy  # noqa: E402

__all__ = ["R", "BondCounter", "PairEntropy", "Trajectory", "adjacency_statistics", "bethe_quasichemical",
           "bond_statistics", "chemical_excess", "density_of_states", "excess_entropy", "first_minimum",
           "fit_redlich_kister", "harmonic_entropy", "ideal_mixing", "lattice", "mutual_information",
           "neighbour_graph", "pair_entropy", "pair_entropy_x_reference", "partial_rdfs", "permutation_check",
           "rdf_cutoffs", "read_ase", "read_lammps_dump", "read_trajectory", "shannon", "vibrational_entropy",
           "__version__"]
