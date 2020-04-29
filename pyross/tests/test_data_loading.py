#!/usr/bin/env python
#
# Tests for data loading (designed to run with pytest)
#
import pyross
import pytest


def test_contact_matrix_india():
    """
    Test loading the Inda contact matrix.
    """
    c = pyross.contactMatrix.India()
    assert len(c) == 4

    # Check shapes
    ch, cw, cs, co = c
    assert ch.shape == (16, 16)
    assert cw.shape == (16, 16)
    assert cs.shape == (16, 16)
    assert co.shape == (16, 16)

    # Check a value (2nd row, 3d column)
    assert ch[1, 2] == pytest.approx(0.719242014301516)

    # And some more
    assert cw[0, 7] == pytest.approx(0.00555435896857786)
    assert cs[15, 15] == pytest.approx(6.61413811374825E-113)
    assert co[12, 2] == pytest.approx(0.0460203564268584)

