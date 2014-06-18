import pydecode.nlp.dependency_parsing as dep
import pydecode.nlp.tagging as tag
import pydecode.nlp.permutation as perm
import pydecode.nlp.decoding as decoding

dependency_instances = [
    (dep.DependencyDecodingProblem(i, order), dep.DependencyScorer, decoder)
    for i in range(60, 61)
    for order, decoder in [(1, dep.FirstOrderDecoder())]
]


def check_decoding(problem, scorer_class, decoder):
    for i in range(1):
        scorer = scorer_class.random(problem)
        hyp_opt = decoder.decode(problem, scorer)


if __name__ == "__main__":
    for problem, cls, decoder in dependency_instances:
        print check_decoding(problem, cls, decoder)
