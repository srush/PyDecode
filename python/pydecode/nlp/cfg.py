import numpy as np
import itertools
import pydecode.encoder
def cnf_cky(n, N):
    items = np.arange((n * n * N), dtype=np.int64) \
        .reshape([n, n, N])

    encoder = CFGEncoder(n, N)
    labels = encoder.encoder

    chart = pydecode.ChartBuilder(items)

    for i in range(n):
        chart.init(items[i,i])

    for d in range(1, n):
        for i in range(n):
            for A in range(N):
                k = i + d
                if k >= n: continue
                if i == 0 and d == n - 1 and A != 0: continue
                chart.set(items[i, k, A],
                          [[items[i, j, B], items[j+1, k, C]]
                           for j in range(i, k)
                           for B in range(N)
                           for C in range(N)],
                          labels=[labels[i,j,k, A, B, C]
                                  for j in range(i, k)
                                  for B in range(N)
                                  for C in range(N)])
            # chart.set_t(items[i, k],
            #             items[i, i:k],
            #             items[i+1:k+1, k],
            #             labels=labels[i, i:k, k])
    return chart.finish(), encoder


class CFGEncoder(pydecode.encoder.StructuredEncoder):
    def __init__(self, n, N):
        self.n = n
        self.N = N
        shape = (n, n, n, N, N, N)
        super(CFGEncoder, self).__init__(shape)

    def transform_structure(self, chart):
        i, k = 0, self.n-1
        stack = [(i,k)]
        parts = []
        while stack:
            i, k = stack.pop()
            if i == k:continue
            for j in range(i, k):
                if chart[i, j] == -1 or chart[j+1, k] == -1:
                    continue
                assert(chart[j+1, k] != -1)
                parts.append((i, j, k,
                              chart[i,k], chart[i,j], chart[j+1, k]))
                stack.append((i,j))
                stack.append((j+1,k))
        return np.array(parts)


    def from_parts(self, parts):
        chart = np.zeros((self.n, self.n))
        chart.fill(-1)
        for part in parts:
            chart[part[0], part[2]] = part[3]
            if part[0] == part[1]:
                chart[part[0], part[1]] = part[4]
            if part[1]+1 == part[2]:
                chart[part[1]+1, part[2]] = part[5]

        return chart


    def all_structures(self):
        for splits in all_splits(0, self.n-1):
            splits = splits[1:]
            for labels in itertools.product(range(self.N),
                                            repeat=len(splits)):
                chart = np.zeros((self.n, self.n), dtype=np.int32)
                chart.fill(-1)
                chart[0, self.n-1] = 0
                for split, label in zip(splits, labels):
                    chart[split] = label

                yield chart

    def random_structure(self):
        chart = np.zeros((self.n, self.n), dtype=np.int32)
        chart.fill(-1)
        splits = random_splits(0, self.n-1)
        for split in splits:
            chart[split] = np.random.randint(self.N)
        return chart

def random_splits(i, k):
    if i == k: return [(i, i)]
    j = np.random.randint(i, k)
    return [(i, k)] + random_splits(i, j) + random_splits(j+1, k)


def all_splits(i, k):
    if i == k: yield [(i, i)]
    for j in range(i,k):
        for a in all_splits(i, j):
            for b in all_splits(j+1, k):
                yield [(i, k)] + a + b
