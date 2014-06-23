import numpy as np
import pydecode.nlp.dependency_parsing as dep
import pydecode.nlp.tagging as tag

#import pydecode.nlp.permutation as perm
import pydecode.nlp.decoding as decoding

dependency_instances = [
    (dep.DependencyProblem(i), decoder)
    for i in range(2, 6)
    for decoder in \
        [(dep.FirstOrderDecoder()),
         (dep.SecondOrderDecoder())]]


def check_decoding(problem, decoder):
    scores = np.random.random(decoder.output_coder(problem).shape_)
    optimal = decoding.decode_exhaustive(problem,
                                         scores,
                                         decoder.output_coder(problem))
    hyp_opt = decoder.decode(problem, scores)
    print optimal, hyp_opt

    assert(optimal == hyp_opt)

def test_decoding():
    for problem, decoder in dependency_instances:
        yield check_decoding, problem, decoder


tagging_instances = [
    (tag.TaggingProblem(i, [1]+([tagset] * (i-2))+[1]),
     decoder)
    for i in range(4, 8)
    for order, decoder in [(2, tag.BigramTagger())]
    for tagset in range(3,5)
]


def test_tagging():
    for problem, decoder in tagging_instances:
        yield check_decoding, problem, decoder

# perm_instances = [
#     (perm.PermutationProblem(i), perm.PermutationScorer, decoder)
#     for i in range(3, 7)
#     for decoder in [perm.PermutationDecoder("ILP"),
#                     perm.PermutationDecoder("BEAM"),
#                     perm.PermutationDecoder("MULTIDFA"),
#                     perm.PermutationDecoder("BIGDFA"),
#                     perm.PermutationDecoder("CUBE")
#                     ]
# ]


# def test_permutation():
#     for problem, cls, decoder in perm_instances:
#         yield check_decoding, problem, cls, decoder

if __name__ == "__main__":
    for a in test_decoding():
        a[0](*a[1:])
