
.. code:: python

    %load_ext cythonmagic
    import numpy as np
.. code:: python

    %%cython
    import numpy as np
    import time
    cimport cython
    
    @cython.wraparound(False)
    @cython.boundscheck(False) 
    @cython.cdivision(False)
    cdef viterbi_fill_trellis(double [:, ::1] e_scores, double [:, ::1] t_scores, double [:, ::1] trellis, int [:, ::1] path):
        cdef:
            double min_score
            double score
            int min_prev
            double e_score
    
            int cur_label, prev_label
            int  feat_i, i, j
            int word_i = 0
            double feat_val
    
            int n_words = e_scores.shape[0]
            int n_labels = e_scores.shape[1]
    
        for word_i in range(n_words):
            # Current label
            for cur_label in range(n_labels):
                min_score = -1E9
                min_prev = -1
    
                e_score = e_scores[word_i, cur_label]
    
                # Transitions from start state
                if word_i == 0:
                    trellis[word_i, cur_label] = e_score + t_scores[n_labels - 1, cur_label]
                    path[word_i, cur_label] = cur_label
                # Transitions from the rest of the states
                else:
                    for prev_label in range(n_labels):
                        score = e_score + t_scores[cur_label, prev_label] + trellis[word_i-1, prev_label]
    
                        if score >= min_score:
                            min_score = score
                            min_prev = prev_label
                    trellis[word_i, cur_label] = min_score
                    path[word_i, cur_label] = min_prev
    
    def benchmark_viterbi_fill_trellis(int n_words=50, int n_labels=45, int n_rounds=1000, int randomize_each=0):
        cdef:
            # Setup reusable data structures outside benchmark
            double [:, ::1] trellis = np.zeros((n_words, n_labels))
            int [:, ::1] path = np.zeros((n_words, n_labels), dtype=np.int32)
    
            double [:, ::1] e_scores = np.random.random(n_words * n_labels).reshape((n_words, n_labels))
            double [:, ::1] t_scores = np.random.random(n_labels * n_labels).reshape((n_labels, n_labels))
    
            int i
    
        start_time = time.time()
        for i in range(n_rounds):
            if randomize_each:
                e_scores = np.random.random(n_words * n_labels).reshape((n_words, n_labels))
                t_scores = np.random.random(n_labels * n_labels).reshape((n_labels, n_labels))
            viterbi_fill_trellis(e_scores, t_scores, trellis, path)
            
        elapsed = time.time() - start_time
        tokens_per_sec = int((n_rounds * n_words) / elapsed)
        return tokens_per_sec
    
    

.. code:: python

    import pydecode.nlp.tagging as tag
    import pydecode
    
    def benchmark_pydecode(n_words=50, n_labels=45, n_rounds=1000):
        problem = tag.TaggingProblem(n_words, [1]+[n_labels]*(n_words-2)+[1])
        tagger = tag.BigramTagger()
        dp = tagger.dynamic_program(problem)
        chart = np.zeros(len(dp.hypergraph.vertices))
        back_pointers = np.zeros(len(dp.hypergraph.vertices), dtype=np.int32)
        scores = np.random.random(len(dp.hypergraph.edges))
        start_time = time.time()
    
        for i in range(n_rounds):
            pydecode.viterbi(dp.hypergraph, scores,
                             chart=chart, back_pointers=back_pointers)
        elapsed = time.time() - start_time
        tokens_per_sec = int((n_rounds * n_words) / elapsed)
        return tokens_per_sec
    

.. code:: python

    print "50 tokens 45 labels", benchmark_viterbi_fill_trellis(50, 45)
    print "50 tokens 45 labels (pydecode)", benchmark_pydecode(50, 45)
    print ""
    print "50 tokens 12 labels", benchmark_viterbi_fill_trellis(50, 12)
    print "50 tokens 12 labels (pydecode)", benchmark_pydecode(50, 12)

.. parsed-literal::

    50 tokens 45 labels 50594
    50 tokens 45 labels (pydecode) 43898
    
    50 tokens 12 labels 596459
    50 tokens 12 labels (pydecode) 422825


Results
-------

Numbers are from my laptop (2,3 GhZ i7).

Without the ``wraparound``, ``boundscheck``, and ``cdivision``
optimizations:

::

    50 tokens 45 labels                      218,732
    50 tokens 45 labels (randomize each)     185,412

    50 tokens 12 labels                    2,900,585
    50 tokens 12 labels (randomize each)   1,640,424

Including all optimizations:

::

    50 tokens 45 labels                      238,150
    50 tokens 45 labels (randomize each)     199,748

    50 tokens 12 labels                    3,851,731
    50 tokens 12 labels (randomize each)   1,944,327


.. code:: python

    