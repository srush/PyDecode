from pydecode.nlp.dependency_parsing import *
import numpy as np

def test_parses():
    assert not is_spanning(np.array([-1, 2, 3, 1]))
    assert not is_spanning(np.array([-1, 1, 1, 1]))
    assert is_spanning(np.array([-1, 0, 1, 1]))

    assert not is_projective(np.array([-1, 2, 4, 0, 0]))

def test_parsing_encoder():
    encoder = DependencyParsingEncoder(10, 1)
    parse = encoder.random_structure()
    parts = encoder.transform_structure(parse)
    test_structure = encoder.from_parts(parts)
    assert (test_structure == parse).all()

def test_parsing_encoder_all():
    encoder = DependencyParsingEncoder(3, 2)
    for parse in encoder.all_structures():
        print parse
        assert(False)
        parts = encoder.transform_structure(parse)
        test_structure = encoder.from_parts(parts)
        assert (test_structure == parse).all()
