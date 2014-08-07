import dependency_parsing
import tagging


class IndexSet:
    def __init__(self, shape):
        self.shape = shape

    def transform(self, labels):
        return np.array(np.unravel_index(labels, self.shape)).T

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
    if order == 1:
        return dependency_parsing.eisner_first_order(sentence_length + 1)
    elif order == 2:
        return dependency_parsing.eisner_second_order(sentence_length + 1)


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
    raise NotImplementedError()
