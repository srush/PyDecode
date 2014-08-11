from pydecode.nlp.cfg import *
import numpy as np

def test_parsing_encoder():
    encoder = CFGEncoder(10, 1)
    parse = encoder.random_structure()
    print parse
    parts = encoder.transform_structure(parse)
    test_structure = encoder.from_parts(parts)
    print test_structure
    assert (test_structure == parse).all()

def test_parsing_encoder_all():
    encoder = CFGEncoder(3, 2)
    for parse in encoder.all_structures():
        parts = encoder.transform_structure(parse)
        test_structure = encoder.from_parts(parts)
        print parse
        print parts
        print test_structure
        assert (test_structure == parse).all()
