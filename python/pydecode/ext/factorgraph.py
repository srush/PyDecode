import pydecode.hyper as ph
import numpy as np
import itertools

class Factor(object):
    def __init__(self, num_variables, possible_labels):
        """
        Parameters
        ----------
        num_variables : int
           The number of variables associated with the factor.

        possible_labels : list of ints
           The number of labels associated with each variable.

        """
        assert(len(possible_labels) == num_variables)
        self.num_variables = num_variables
        self.possible_labels = possible_labels
        self.max_labels = max(possible_labels)
        self.factor_id = -1

    def size(self):
        return self.num_variables

    def score(self, labels):
        """
        Returns the score \theta_f.

        Parameters
        -----------
        labels : list of labels
            The labels x_f

        Returns
        --------
        The value of \theta_f(x_f)
        """
        pass

    def argmax(self, reparameters):
        """
        Given reparameterization returns the
        argmax assignment.

        Parameters
        ----------
        reparameters : matrix
           Matrix \delta_i,l associated with
           variable i and label l

        Returns
        ---------
        argmax : list of labels
            argmax[i]
            The label assigned to variable i
            in the argmax assignment
        """
        pass

    def max_marginals(self, reparameters):
        """
        Given reparameterization returns the
        max-marginal value for each variable taking
        a given label.

        Parameters
        ----------
        reparameters : matrix
           Matrix \delta_i,l associated with
           variable i and label l

        Returns
        ---------
        maxmarginals : matrix
           mm[i][j]
           The maximum value of this factor when
           variable i takes on label j.
        """
        pass

class HypergraphFactor(Factor):
    def __init__(self, num_variables, possible_labels,
                 hypergraph, weights, labels):
        self.hypergraph = hypergraph
        self.weights = weights
        self.labels = labels
        super(HypergraphFactor, self).__init__(
            num_variables, possible_labels)

    def score(self, labels):
        s = set(enumerate(labels))
        score = 0.0

        binary = ph.BoolPotentials(self.hypergraph)\
            .from_vector([1 if all((l in s for l in self.labels[edge])) else 0
                          for edge in self.hypergraph.edges])

        path = ph.best_path(self.hypergraph, binary)
            # if all((l in s for l in self.labels[edge])):
            #     score += self.weights[edge]
            #     for l in self.labels[edge]:
            #         s.remove(l)

        if len(path.edges) ==  0:
            return -1e9
        return self.weights.dot(path)

    def _reparam(self, reparams):
        pot = ph.LogViterbiPotentials(self.hypergraph)\
            .from_vector([sum((reparams[l]
                               for l in self.labels[edge]))
                          for edge in self.hypergraph.edges])
        return self.weights.times(pot)


    def argmax(self, reparams):
        new_pot = self._reparam(reparams)
        path = ph.best_path(self.hypergraph, new_pot)

        labels = self.labels.dot(path)
        # for edge in path.edges:
        #     print edge, edge.head.label, edge.tail[0].label, self.labels[edge], "|",
        # # print
        # print labels
        argmax = [-1] * self.num_variables
        for i, l in labels:
            argmax[i] = l
        # print argmax
        assert(-1 not in argmax)
        # print argmax
        return argmax

    def max_marginals(self, reparams):
        new_pot = self._reparam(reparams)
        # print [self.weights[edge] for edge in self.hypergraph.edges]
        # print [new_pot[edge] for edge in self.hypergraph.edges]
        marginals = ph.compute_marginals(self.hypergraph, new_pot)
        mm = np.zeros((self.num_variables, self.max_labels))

        best = [-1e9] * self.num_variables
        for i in range(self.num_variables):
            for l in range(self.max_labels):
                best[i] = max(best[i], reparams[i, l])
        total_best = sum(best)

        for i in range(self.num_variables):
            for l in range(self.max_labels):
                mm[i,l] = -1e9 + total_best - best[i] + reparams[i,l]

        for edge in self.hypergraph.edges:
            for i, l in self.labels[edge]:
                mm[i, l] = max(marginals[edge], mm[i, l])
        return mm

class AgreeFactor(Factor):
    """
    Enforces that all the variables given agree
    with each other.
    """
    def __init__(self, num_variables, possible_labels):
        super(AgreeFactor, self).__init__(num_variables, possible_labels)

    def score(self, labels):
        for a, b in itertools.izip(labels, labels[1:]):
            if a != b: return -1e9
        return 0

    def _score_agree(self, reparams, label):
        return sum((reparams[v, label] for v in range(self.num_variables)))

    def argmax(self, reparams):
        best = -1e9
        for l in range(self.possible_labels[0]):
            s = self._score_agree(reparams, l)
            if s > best:
                best = s
                labels = [l] * self.num_variables
        return labels

    def max_marginals(self, reparams):
        mm = np.zeros((self.num_variables, self.max_labels))
        scores = [self._score_agree(reparams, l)
                 for l in range(self.possible_labels[0])]

        for i in range(self.num_variables):
            for l in range(self.possible_labels[i]):
                mm[i, l] = scores[l]

        return mm

class TabularFactor(Factor):
    def __init__(self, num_variables, possible_labels, score_table):
        # Assuming a matrix (only two variables max).
        self.score_table = score_table
        super(TabularFactor, self).__init__(num_variables, possible_labels)

    def score(self, labels):
        assert(len(labels) == 2)
        return self.score_table[labels[0]][labels[1]]

    def argmax(self, reparams):
        best = -1e9
        for l in range(self.possible_labels[0]):
            for l2 in range(self.possible_labels[1]):
                s = self.score_table[l][l2] + reparams[0, l] +  reparams[1, l2]
                if s > best:
                    best = s
                    labels = [l, l2]
        return labels

    def max_marginals(self, reparams):
        mm = np.zeros((self.num_variables, self.max_labels))
        for l in range(self.possible_labels[0]):
            mm[0, l] = max((self.score_table[l][l2] + reparams[0, l] +  reparams[1, l2]
                            for l2 in range(self.possible_labels[1])))

        for l in range(self.possible_labels[1]):
            mm[1, l] = max((self.score_table[l2][l] + reparams[1, l] +  reparams[0, l2]
                            for l2 in range(self.possible_labels[0])))
        return mm


def exhaustive_search(factor_graph):
    assignments = []
    b = -1e9
    l = None
    for labeling in itertools.product(*[range(m) for m in factor_graph.possible_labels]):
        labeling = list(labeling)
        s = factor_graph.score(labeling)
        if s > b:
            l = labeling
            b = s
    return l, b


def exhaustive_max_marginals(factor_graph, factor):
    reparam = np.random.random((factor.num_variables, factor.max_labels))
    mm = factor.max_marginals(reparam)

    mm2 = np.zeros((factor.num_variables, factor.max_labels))
    mm2.fill(-1e9)

    for labeling in itertools.product(*[range(m) for m in factor_graph.possible_labels]):
        labeling = list(labeling)
        factor_labels = factor_graph.factor_labeling(factor, labeling)
        score = factor.score(factor_labels)
        for i in range(factor.num_variables):
            score += reparam[i, factor_labels[i]]

        for i in range(factor.num_variables):
            mm2[i, factor_labels[i]] = max(mm2[i, factor_labels[i]], score)

    # print mm
    # print mm2

    # print ((abs(mm - mm2)) < 1e-4)
    assert(((abs(mm - mm2)) < 1e-4).all())
    return mm2


class FactorGraph:
    def __init__(self, num_variables, possible_labels, unary_scores):
        self.num_variables = num_variables
        self.possible_labels = possible_labels
        self.factors = []
        self.factor_variables = []
        self.unary_scores = unary_scores
        self.max_labels = max(self.possible_labels)
        for v in range(num_variables):
            assert(len(unary_scores[v]) == possible_labels[v])

        self.local_names = {}

    def local_name(self, factor_id, global_var_id):
        return self.local_names[factor_id, global_var_id]

    def has_variable(self, factor_id, global_var_id):
        return (factor_id, global_var_id) in self.local_names

    def factor_labeling(self, factor, labeling):
        assert(factor.id != -1)
        return [labeling[v] for v in self.factor_variables[factor.id]]

    def argmax(self, reparams):
        argmax = [-1] * self.num_variables
        score = 0.0
        for v in range(self.num_variables):
            best = -1e9
            for l in range(self.possible_labels[v]):

                s = self.unary_scores[v][l] + reparams[v, l]
                #print v, l , s,
                if s > best:
                    best = s
                    argmax[v] = l
            score += best

        return argmax, score



    def score(self, labels):
        """
        Given labels of all x returns the score, i.e.
        the sum of the factor scores plus the unary scores.


        Parameters
        -----------
        labels : list of labels
            The labels x

        Returns
        --------
        Sum of \sum_i \theta_i(x_i) + \sum_f \theta_f(x_f).
        """
        score = 0
        for i in range(self.num_variables):
            score += self.unary_scores[i][labels[i]]
            # if reparams is not None:
            #     score += reparams[i][labels[i]]

        for j, f in enumerate(self.factors):
            score += f.score([labels[v] for v in self.factor_variables[j]])
            # if reparams is not None:
            #     score -= reparams[i][labels[i]]

        return score

    def register_factor(self, factor, variables):
        """
        Parameters
        -----------
        factor : Factor

        variables : list of int's
            The variables the factor score.
        """
        factor_id = len(self.factors)
        self.factors.append(factor)
        self.factor_variables.append(variables)
        for i, var in enumerate(variables):
            self.local_names[factor_id, var] = i
        factor.id = factor_id

def get_reparams_variables(delta, factor_graph):
    reparams = np.zeros((factor_graph.num_variables, factor_graph.max_labels))
    for j, f in enumerate(factor_graph.factors):
        for i, v in enumerate(factor_graph.factor_variables[j]):
            for l in range(factor_graph.possible_labels[v]):
                reparams[v, l] += delta[j][i, l]
    return reparams

def get_reparams_factor(delta, factor_graph, f, j):
    reparams = np.zeros((f.num_variables, f.max_labels))
    for i, v in enumerate(factor_graph.factor_variables[j]):
        for l in range(factor_graph.possible_labels[v]):
            # \delta^-1
            reparams[i, l] = \
                factor_graph.unary_scores[v][l] + \
                sum((delta[j2][factor_graph.local_name(j2, v), l]
                     for j2, f2 in enumerate(factor_graph.factors)
                     if factor_graph.has_variable(j2, v)
                     if j2 != j))
    return reparams

def mplp(factor_graph, iterations=100):
    delta = []
    for j, f in enumerate(factor_graph.factors):
        delta.append(np.zeros((f.num_variables, f.max_labels)))

    for q in range(iterations):
        for j, f in enumerate(factor_graph.factors):
            reparams = get_reparams_factor(delta, factor_graph, f, j)
            mm = f.max_marginals(reparams)
            label = f.argmax(reparams)
            delta[j] = -reparams + \
                (1.0 / float(f.size())) * mm

        # Best Assignment
        reparams = get_reparams_variables(delta, factor_graph)

        labeling, score = factor_graph.argmax(reparams)
        print labeling, score
        print factor_graph.score(labeling)
        # print factor_graph.score(labeling, reparams)
