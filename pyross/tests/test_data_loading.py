#!/usr/bin/env python
import pyross


def test_data_loading_india():
    cs = pyross.contactMatrix.India()
    assert len(cs) == 4

