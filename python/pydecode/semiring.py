"""
THIS CLASS IS DEPRECATED. CODE MOVED TO C++.
"""


INF = 1e8


class SemiRing(object):
    """
    A semiring operation.

    Implements + and *

    """
    def __add__(self, other):
        raise NotImplementedError()

    def __mul__(self, other):
        raise NotImplementedError()

    @classmethod
    def one(cls):
        raise NotImplementedError()

    @classmethod
    def zero(cls):
        raise NotImplementedError()

    @classmethod
    def make(cls, v):
        raise NotImplementedError()


class LogicSemiRing(SemiRing):
    """

    """

    def __init__(self, v):
        self.v = v

    def __repr__(self):
        return self.v.__repr__()

    def is_zero(self):
        return not self.v

    def __add__(self, other):
        return LogicSemiRing(self.v or other.v)

    def __mul__(self, other):
        return LogicSemiRing(self.v and other.v)

    def unpack(self):
        return self.v

    @classmethod
    def one(cls):
        return LogicSemiRing(True)

    @classmethod
    def zero(cls):
        return LogicSemiRing(False)

    @classmethod
    def make(cls, v):
        return LogicSemiRing(v)

class ViterbiSemiRing(SemiRing):
    """
    The viterbi max semiring.

    """

    def __init__(self, v):
        self.v = v

    def __repr__(self):
        return self.v.__repr__()

    def is_zero(self):
        return self.v <= -INF

    def __add__(self, other):
        return ViterbiSemiRing(max(self.v, other.v))

    def __mul__(self, other):
        return ViterbiSemiRing(self.v + other.v)

    def unpack(self):
        return self.v

    @classmethod
    def one(cls):
        return ViterbiSemiRing(0.0)

    @classmethod
    def zero(cls):
        return ViterbiSemiRing(-INF)

    @classmethod
    def make(cls, v):
        return ViterbiSemiRing(v)


class ProbSemiRing(SemiRing):
    def __init__(self, v):
        self.v = v

    def __repr__(self):
        return self.v.__repr__()

    def is_zero(self):
        return self.v == 0.0

    def __add__(self, other):
        return ProbSemiRing(max(self.v, other.v))

    def __mul__(self, other):
        return ProbSemiRing(self.v + other.v)

    def unpack(self):
        return self.v

    @classmethod
    def one(cls):
        return ProbSemiRing(1.0)

    @classmethod
    def zero(cls):
        return ProbSemiRing(0.0)

    @classmethod
    def make(cls, v):
        return ProbSemiRing(v)


class HypergraphSemiRing(SemiRing):
    def __init__(self, edge_list=[], node_list=[],
                 name=None, is_zero=False):
        self.edge_list = edge_list
        self.node_list = node_list
        self.name = name
        self._is_zero = is_zero

    def __repr__(self):
        return "%s %s %s %s" % (self.edge_list,
                                self.node_list,
                                self.name,
                                self._is_zero)

    def is_zero(self):
        return self._is_zero

    def __add__(self, other):
        return HypergraphSemiRing(
            self.edges() + other.edges())

    def edges(self):
        if self.node_list:
            return self.edge_list + [(self.node_list, self.name)]
        else:
            return self.edge_list

    def __mul__(self, other):
        zero = other.is_zero() or self.is_zero()
        if zero:
            return HypergraphSemiRing.zero()
        return HypergraphSemiRing([],
                                  self.node_list + other.node_list,
                                  other.name)

    @classmethod
    def one(cls):
        return HypergraphSemiRing()

    @classmethod
    def zero(cls):
        return HypergraphSemiRing(is_zero=True)

    @classmethod
    def make(cls, name):

        return HypergraphSemiRing(name=name)
