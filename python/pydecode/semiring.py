INF = 1e8
class SemiRing:
    """
    A semiring operation.

    Implements + and *

    """
    def __add__(self, other):
        raise NotImplementedError()

    def __mul__(self, other):
        raise NotImplementedError()

    @staticmethod
    def one(): raise NotImplementedError()

    @staticmethod
    def zero(): raise NotImplementedError()

    @staticmethod
    def make(v): raise NotImplementedError()



class ViterbiSemiRing(SemiRing):
    """
    The viterbi max semiring.

    """

    def __init__(self, v): self.v = v

    def __repr__(self): return self.v.__repr__()

    def is_zero(self): return self.v <= -INF

    def __add__(self, other):
        return ViterbiSemiRing(max(self.v, other.v))

    def __mul__(self, other):
        return ViterbiSemiRing(self.v + other.v)

    def unpack(self): return self.v

    @staticmethod
    def one(): return ViterbiSemiRing(0.0)

    @staticmethod
    def zero(): return ViterbiSemiRing(-INF)

    @staticmethod
    def make(v): return ViterbiSemiRing(v)

class ProbSemiRing(SemiRing):
    def __init__(self, v): self.v = v

    def __repr__(self): return self.v.__repr__()

    def is_zero(self): return self.v == 0.0

    def __add__(self, other):
        return ProbSemiRing(max(self.v, other.v))

    def __mul__(self, other):
        return ProbSemiRing(self.v + other.v)

    def unpack(self): return self.v

    @staticmethod
    def one(): return ProbSemiRing(1.0)

    @staticmethod
    def zero(): return ProbSemiRing(0.0)

    @staticmethod
    def make(v): return ProbSemiRing(v)

class HypergraphSemiRing(SemiRing):
    def __init__(self, edge_list=[], node_list=[],
                 name=None, is_zero=False):
        self.edge_list = edge_list
        self.node_list = node_list
        self.name = name
        self._is_zero = is_zero

    def is_zero(self): return self._is_zero



    def __add__(self, other):
        return HypergraphSemiRing(self.edge_list + [(other.node_list, other.name)],
                                  [], None)

    def __mul__(self, other):
        zero = other.is_zero() or self.is_zero()
        if zero: HypergraphSemiRing.zero()
        return HypergraphSemiRing([],
                                  self.node_list + other.node_list,
                                  other.name)

    @staticmethod
    def one(): return HypergraphSemiRing()

    @staticmethod
    def zero(): return HypergraphSemiRing(is_zero=True)

    @staticmethod
    def make(name): return HypergraphSemiRing([], [], name)
