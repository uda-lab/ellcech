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
from .minimax import MinimaxResult, solve_minimax, solve_minimax_from_coefs

__all__ = [
    "__version__",
    "MinimaxResult",
    "solve_minimax",
    "solve_minimax_from_coefs",
]
