"""ellphi-alpha: Experimental anisotropic alpha complex for ellipsoid-based TDA.

This package implements the filtration value computation for the anisotropic
alpha complex, whose mathematical foundations are formalised in Lean 4 in the
LeanEllAlpha project (AnisotropicKKT.lean, Duality.lean).

Relation to ellphi
------------------
ellphi computes pairwise tangency distances and feeds them into Vietoris-Rips.
ellphi-alpha computes alpha-complex filtration values α(σ) for simplices σ of
any dimension, enabling a sparser (Delaunay-like) filtration.

Value correspondence (pairwise case |σ| = 2):
    alpha_result.alpha  ==  ellphi.tangency(p, q).t ** 2
"""

from ._version import __version__
from .act_map import (
    BooleanityReport,
    SimplexInfo,
    SupportCertificate,
    act_support,
    booleanity_report,
    certify_support,
    coface_set,
    critical_simplices,
    critical_value_check,
    fiber_partition,
)
from .enumeration import enumerate_candidate_simplices, iter_candidate_simplices
from .filtration import FiltrationEntry, build_incremental_filtration
from .gudhi_bridge import to_gudhi_simplex_tree
from .minimax import MinimaxResult, solve_minimax, solve_minimax_from_coefs
from .minimax_grad import GradientResult, compute_gradient
from .predicates import PredicateResult, evaluate_predicates
from .pruning import (
    CertifiedFiltration,
    PruningStats,
    brute_force_filtration,
    certified_filtration,
    clique_candidates,
    neighbour_graph,
    rayleigh_lower_bound,
)

__all__ = [
    "__version__",
    "enumerate_candidate_simplices",
    "iter_candidate_simplices",
    "MinimaxResult",
    "solve_minimax",
    "solve_minimax_from_coefs",
    "GradientResult",
    "compute_gradient",
    "PredicateResult",
    "evaluate_predicates",
    "FiltrationEntry",
    "build_incremental_filtration",
    "to_gudhi_simplex_tree",
    "SupportCertificate",
    "SimplexInfo",
    "BooleanityReport",
    "certify_support",
    "act_support",
    "coface_set",
    "fiber_partition",
    "booleanity_report",
    "critical_simplices",
    "critical_value_check",
    "CertifiedFiltration",
    "PruningStats",
    "rayleigh_lower_bound",
    "neighbour_graph",
    "clique_candidates",
    "certified_filtration",
    "brute_force_filtration",
]
