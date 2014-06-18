import pydecode.nlp.dependency_parsing as dep
import pydecode.nlp.tagging as tag
import pydecode.nlp.permutation as perm
import pydecode.nlp.decoding as decoding

dependency_instances = [
    (dep.DependencyDecodingProblem(i, order), dep.DependencyScorer, decoder)
    for i in range(2, 6)
    for order, decoder in [(1, dep.FirstOrderDecoder())]

]


def check_decoding(problem, scorer_class, decoder):
    scorer = scorer_class.random(problem)
    exhaustive = decoding.ExhaustiveDecoder()
    optimal = exhaustive.decode(problem, scorer)
    hyp_opt = decoder.decode(problem, scorer)
    print optimal, hyp_opt
    assert(optimal == hyp_opt)

def test_decoding():
    for problem, cls, decoder in dependency_instances:
        yield check_decoding, problem, cls, decoder


tagging_instances = [
    (tag.TaggingProblem(i, order, tagset), tag.TagScorer, decoder)
    for i in range(2, 7)
    for order, decoder in [(2, tag.BigramTagger()), (3, tag.TrigramTagger())]
    for tagset in [range(5)]
]

#  @ignoretest
# def test_tagging():
#     for problem, cls, decoder in tagging_instances:
#         yield check_decoding, problem, cls, decoder

perm_instances = [
    (perm.PermutationProblem(i), perm.PermutationScorer, decoder)
    for i in range(3, 7)
    for decoder in [perm.PermutationDecoder("ILP"),
                    perm.PermutationDecoder("BEAM"),
                    perm.PermutationDecoder("MULTIDFA"),
                    perm.PermutationDecoder("BIGDFA"),
                    perm.PermutationDecoder("CUBE")
                    ]
]


def test_permutation():
    for problem, cls, decoder in perm_instances:
        yield check_decoding, problem, cls, decoder

if __name__ == "__main__":
    for a in test_decoding():
        a[0](*a[1:])
