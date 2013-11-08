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


class HypergraphSemiRing(SemiRing):
    def __init__(self, name=None, edge_list=[], node_list=[], is_zero=False):
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
        return HypergraphSemiRing(name = None,
            edge_list=(self.edges() + other.edges()))

    def edges(self):
        if self.node_list:
            return self.edge_list + [(self.node_list, self.name)]
        else:
            return self.edge_list

    def __mul__(self, other):
        zero = other.is_zero() or self.is_zero()
        if zero:
            return HypergraphSemiRing.zero()
        return HypergraphSemiRing(other.name, [],
                                  self.node_list + other.node_list,
                                  )

    @classmethod
    def one(cls):
        return HypergraphSemiRing()

    @classmethod
    def zero(cls):
        return HypergraphSemiRing(is_zero=True)

    @classmethod
    def make(cls, name):
        return HypergraphSemiRing(name=name)
