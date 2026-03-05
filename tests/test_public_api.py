import ellphi_alpha as ea


def test_public_api_exports_new_symbols():
    assert callable(ea.iter_candidate_simplices)
    assert callable(ea.enumerate_candidate_simplices)
    assert callable(ea.evaluate_predicates)
    assert callable(ea.build_incremental_filtration)
    assert callable(ea.to_gudhi_simplex_tree)
    assert ea.PredicateResult is not None
    assert ea.FiltrationEntry is not None
