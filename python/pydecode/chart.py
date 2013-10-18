class ViterbiSemiRing:
    def __init__(self, v):
        self.v = v

    def __repr__(self):
        return self.v.__repr__()

    @staticmethod
    def one():
        return ViterbiSemiRing(0.0)

    @staticmethod
    def zero():
        return ViterbiSemiRing(-100000)

    def is_zero(self):
        return self.v == -100000

    @staticmethod
    def make(v):
        return ViterbiSemiRing(v)

    def __add__(self, other):
        return ViterbiSemiRing(max(self.v, other.v))

    def __mul__(self, other):
        return ViterbiSemiRing(self.v + other.v)

class HypergraphSemiRing:
    def __init__(self, edge_list=[], node_list=[],
                 name=None, is_zero=False):
        self.edge_list = edge_list
        self.node_list = node_list
        self.name = name
        self._is_zero = is_zero

    @staticmethod
    def one():
        return HypergraphSemiRing()

    @staticmethod
    def zero():
        return HypergraphSemiRing(is_zero=True)

    def is_zero(self):
        return self._is_zero

    @staticmethod
    def make(name):
        return HypergraphSemiRing([], [], name)

    def __add__(self, other):
        return HypergraphSemiRing(self.edge_list + [(other.node_list, other.name)],
                                  [], None)

    def __mul__(self, other):
        zero = other.is_zero() or self.is_zero()
        if zero: HypergraphSemiRing.zero()
        return HypergraphSemiRing([],
                                  self.node_list + other.node_list,
                                  other.name)

class ChartBuilder:
    def __init__(self,
                 scorer,
                 builder=None,
                 semiring=ViterbiSemiRing):
        self.builder = builder
        self.chart = {}
        self.semiring = semiring
        self.scorer = scorer

    def sr(self, label):
        return self.semiring.make(self.scorer(label))

    def init(self, label):
        if self.builder is not None:
            print "start"
            node = self.builder.add_node([], label=label)
            self.chart[label] = HypergraphSemiRing([], [node], None)
        else:
            self.chart[label] = self.semiring.one()


    def __setitem__(self, label, val):
        if label in self.chart:
            raise Exception(
                "Chart already has label {}".format(label))
        print label, val
        if self.builder is not None:
            if not val.is_zero():
                print val.edge_list
                node = self.builder.add_node(val.edge_list,
                                             label=label)
                self.chart[label] = \
                    HypergraphSemiRing([], [node], None)
        else:
            if not val.is_zero():
                self.chart[label] = val

    def sum(self, edges):
        return sum(edges, self.semiring.zero())

    def __contains__(self, label):
        return label in self.chart

    def __getitem__(self, label):
        return self.chart.get(label, self.semiring.zero())
