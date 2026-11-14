"""The examples of the docstrings (as R CMD check runs the examples of the help pages)."""
import doctest

import pytest

from dflimit import _adfmax, _coef, _lincdf, _ltratio, _region, _threads


@pytest.mark.parametrize("mod", [_ltratio, _coef, _adfmax, _region, _lincdf, _threads], ids=lambda m: m.__name__)
def test_examples(mod):
    r = doctest.testmod(mod, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE, verbose=False)
    assert r.attempted > 0
    assert r.failed == 0
