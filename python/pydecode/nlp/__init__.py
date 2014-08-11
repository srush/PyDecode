import dependency_parsing
import tagging
import cfg as cfg_
import numpy as np
from collections import Counter, defaultdict
import re
import numpy as np


def eisner(sentence_length, order=1):
    """
    Implements the dynamic programming algorithm
    for projective dependency parsing.

    Parameters
    ----------
    sentence_length : int
        The length of the sentence.

    order : int
        The order of dependency arcs. Currently
        implements {1,2}

    Returns
    -------
    graph : :py:class:`Hypergraph`
        Hypergraph encoding all valid parses.

    index_set :

    """
    n = sentence_length + 1
    if order == 1:
        return dependency_parsing.eisner_first_order(n)
    elif order == 2:
        return dependency_parsing.eisner_second_order(n)


def tagger(sentence_length, tag_sizes, order=1):
    """
    Implements dynamic programming algorithm
    for an ngram tagger.

    Parameters
    ----------
    sentence_length : int
        The length of the sentence.

    order : int
        The order of the tagger. Currently
        implements {1}.

    Returns
    -------
    graph : :py:class:`Hypergraph`
        Hypergraph encoding all

    index_set :
    """
    if order == 1:
        return tagging.tagger_first_order(sentence_length, tag_sizes)

def semimarkov(sentence_length):
    """
    Implements dynamic programming algorithm
    for a semi-markov tagger.

    Parameters
    ----------
    sentence_length : int
        The length of the sentence.


    Returns
    -------
    graph : :py:class:`Hypergraph`
        Hypergraph encoding all

    index_set :
    """
    raise NotImplementedError()

def cfg(sentence_length, grammar_size):
    """
    Implements dynamic programming algorithm
    for a Chomsky normal form.

    Parameters
    ----------
    sentence_length : int
        The length of the sentence.


    Returns
    -------
    graph : :py:class:`Hypergraph`
        Hypergraph encoding all

    index_set :
    """
    return cfg_.cnf_cky(sentence_length, grammar_size)


def read_csv_records(f, front=[], back =[], limit=None, length=None):
    s = open(f).read()
    for i, l in enumerate(re.finditer("(.*?)\n\n", s, re.DOTALL)):
        if limit is not None and i > limit:
            break
        if length is not None and len(l.group(1).split("\n")) > length:
            continue
        yield np.array(front + [line.split()
                        for line in l.group(1).split("\n")] + back)

CONLL = {"INDEX":1,
         "WORD":1,
         "TAG":3,
         "HEAD":6,
         "LABEL":7}

TAG = {"WORD":0,
       "TAG":1}
