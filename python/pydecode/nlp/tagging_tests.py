from pydecode.nlp.tagging import *
import numpy as np

def test_tagging():
    encoder = TaggingEncoder([5] * 10, 1)
    sequence = encoder.random_structure()
    parts = encoder.transform_structure(sequence)
    test_structure = encoder.from_parts(parts)
    assert (test_structure == sequence).all()
