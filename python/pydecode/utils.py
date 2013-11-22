

def all_paths(h):
    """
    Gets all possible hyperpaths.
    
    """
    def paths(node):
        if node.is_terminal:
            yield tuple()
        else:
            for edge in node.edges:
                t = [paths(node) for node in edge.tail]
                for below in itertools.product(*t):
                    yield (edge,) + sum(below, ())
    paths = [ph.Path(h, list(edges)) for edges in paths(h.root)]
    return paths

def random_path(h):
    """
    
    """

    raise Exception("Not yet implemented")
