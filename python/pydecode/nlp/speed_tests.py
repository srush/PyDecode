import pydecode
import time
import pydecode.nlp.dependency_parsing as dep
import pydecode.nlp.tagging as tag
#import pydecode.nlp.permutation as perm
import pydecode.nlp.decoding as decoding
import numpy as np

BIG = 45

def make_parsing_dp():
    problem = dep.DependencyProblem(60)
    parser = dep.FirstOrderDecoder()
    scores = np.random.random(parser.output_coder(problem).shape_)
    dp = parser.dynamic_program(problem)
    return dp, scores

def make_tagging_dp():
    problem = tag.TaggingProblem(BIG, [1]+[BIG]*(BIG-2)+[1])
    tagger = tag.BigramTagger()
    scores = np.random.random(tagger.output_coder(problem).shape_)
    dp = tagger.dynamic_program(problem)
    return dp, scores

def time_argmax(dp, scores):
    chart = np.zeros(len(dp.hypergraph.edges))
    for _ in range(1000):
        pydecode.argmax(dp, scores, chart=chart)

def time_argmax_masked(dp, scores):
    chart = np.zeros(len(dp.hypergraph.edges))
    mask = np.array(np.random.choice([0, 0, 0, 0, 1],
                                     dp.items.shape),
                    dtype=np.uint8)

    for _ in range(1000):
        pydecode.argmax(dp, scores, chart=chart, mask=mask)

def time_hypergraph(dp, scores):
    chart = np.zeros(len(dp.hypergraph.vertices))
    for _ in range(1000):
        pydecode.best_path(dp.hypergraph, scores,
                           chart=chart)

def time_hypergraph_masked(dp, scores):
    chart = np.zeros(len(dp.hypergraph.vertices))
    mask = np.array(np.random.choice([1] * 2,
                                     len(dp.hypergraph.vertices)),
                    dtype=np.uint8)
    # print  np.sum(mask) / float(len(mask))

    for _ in range(1000):
        pydecode.best_path(dp.hypergraph, scores,
                           chart=chart, mask=mask)
        # print np.sum(mask) / float(len(mask))
def time_trellis():
    n_labels = BIG
    words = np.array([1] * BIG, dtype=np.int32)
    trellis = np.zeros([n_labels+2, n_labels+2], dtype=np.float32)
    path = np.zeros([n_labels+2, n_labels+2], dtype=np.int32)
    emissions = np.array(np.random.random([BIG, BIG]), dtype=np.float32)

    transitions = np.array(np.random.random([n_labels+2, n_labels+2]), dtype=np.float32).ravel()

    # transitions = np.array(np.random.choice([-1e9] * 1 + [1] * 1,
    #                  [n_labels+2, n_labels+2]), dtype=np.float32).ravel()
    # print transitions
    # transitions.fill(-1e9)
    # transitions = np.array(np.random.choice([-1e9] * 2 + [1] * 1,
    #                                         [n_labels+2, n_labels+2]), dtype=np.float32).ravel()

    for _ in range(1000):
        pydecode.fill_trel(emissions, transitions, n_labels, words, trellis, path)

def main():
    dp, scores = make_tagging_dp()
    # s = time.time()
    # time_argmax(dp, scores)
    # print ((1000 * 50) / (time.time() - s))

    s = time.time()
    time_trellis()
    print ((1000 * 50) / (time.time() - s))

    hypergraph_scores = np.random.choice([-1e10] * 10 + [1] * 5,
                                         len(dp.hypergraph.edges))
    s = time.time()
    # transitions = np.array(np.random.choice([-1e9] * 1 + [1] * 1,
    #                  [n_labels+2, n_labels+2]), dtype=np.float32).ravel()

    time_hypergraph(dp, hypergraph_scores)
    print ((1000 * 50) / (time.time() - s))

    # s = time.time()
    # time_hypergraph_masked(dp, hypergraph_scores)
    # print time.time() - s


if __name__ == "__main__":
    main()

# time 1000 with 11.93
