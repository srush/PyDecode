from distutils.core import setup
from distutils.extension import Extension
import os.path
import sys
import numpy as np

def check_for_cython():
    return True

class ExtensionWrapper:
    def __init__(self, debug=False, cython=False):
        self.debug = debug
        self.cython = cython

    def make(self, ext_name, pyx_name, cpp_names,
             extra_objects=[]):
        return Extension(ext_name,
                         [pyx_name] + cpp_names
                         if self.cython
                         else [pyx_name.split(".")[0] + "." + "cpp"] + cpp_names,
                         language='c++',
                         extra_compile_args=["-O0"] if self.debug else [],
                         include_dirs=[r'src/', "."])

    def cmdclass(self):
        if self.cython:
            from Cython.Distutils import build_ext
            from Cython.Build import cythonize
            return {'build_ext': build_ext}
        return {}


def make_extension(wrapper):
    a = [wrapper.make("pydecode._pydecode",
                     "python/pydecode/_pydecode.pyx",
                     ["src/Hypergraph/Hypergraph.cpp",
                      "src/Hypergraph/Semirings.cpp",
                      "src/Hypergraph/SemiringAlgorithms.cpp",
                      "src/Hypergraph/Algorithms.cpp",
                      "src/Hypergraph/BeamSearch.cpp"
                      ])]
    return a

def main():
    copy_args = sys.argv[1:]
    has_cython = check_for_cython()
    if '--cython' not in copy_args:
        has_cython = False
    if '--cython' in copy_args:
        copy_args.remove("--cython")
    debug = False
    if '--debug' in copy_args:
        debug = True
        copy_args.remove("--debug")

    print sys.argv
    print "done"
    wrapper = ExtensionWrapper(cython=has_cython, debug=debug)

    setup(
        name = 'pydecode',
        cmdclass = wrapper.cmdclass(),
        packages=['pydecode', 'pydecode.nlp', 'pydecode.test'],
        package_dir={'pydecode': 'python/pydecode'},
        ext_modules = make_extension(wrapper),
        requires=["numpy"],
        version = '0.2.11',
        description = 'A dynamic programming toolkit',
        author = 'Alexander Rush',
        author_email = 'srush@csail.mit.edu',
        url = 'https://github.com/srush/pydecode/',
        download_url = 'https://github.com/srush/PyDecode/tarball/master',
        keywords = ['nlp'],
        classifiers = [],
        script_args = copy_args,
        include_dirs = [np.get_include()],
        data_files=[('pydecode/notebooks', ['notebooks/hmm.ipynb',
                                            "notebooks/Fibonacci.ipynb",
                                            "notebooks/EditDistance.ipynb",
                                            "notebooks/parsing.ipynb"]
                                        )]
        )

if __name__ == "__main__":
    main()
