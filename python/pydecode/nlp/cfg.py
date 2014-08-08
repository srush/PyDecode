def cnf_cky(n, N):
    items = np.arange((n * n * N), dtype=np.int64) \
        .reshape([n, n, N])
    labels = np.arange(n*n*n*N*N*N, dtype=np.int64).reshape([n, n, n, N, N, N])

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
    return chart.finish()


class CFGEncoder:
    def __init__(self, n, N):
        self.n = n
        self.N = N
        self.shape = (n, n, n, N, N, N)

    def transform(self, labels):
        return np.array(np.unravel_index(labels, self.shape)).T

    def from_path(self, path):
        parse = self.transform(path.labeling[path.labeling!=-1])
        return self.from_labels(parse)

    # def to_labels(self, tagging):
    #     return np.array([[i] + tagging[i-self.order:i+1]
    #                      for i in range(self.order, len(tagging))])

    # def from_labels(self, labels):
    #     sequence = np.zeros(self.size)
    #     for (i, pt, t) in labels:
    #         sequence[i] = t
    #         sequence[i-1] = pt
    #     return sequence
