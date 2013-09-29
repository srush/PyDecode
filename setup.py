from distutils.core import setup
from distutils.extension import Extension

try:
    from Cython.Distutils import build_ext
    from Cython.Compiler.Version import version
    if int(version.split(".")[1]) < 19:
        raise ImportError("Bad version.")
    
except ImportError:
    use_cython = False
else:
    use_cython = True


ext_modules = [ ]

if use_cython:
    ext_modules = [Extension("pydecode.hyper",
                             ["python/pydecode/hyper.pyx"],
                             language='c++',
                             include_dirs=[r'build/debug/src/', "."],
                             library_dirs=[r'build/debug/src/', ""],
                             extra_objects=['build/debug/src/libdecoding.a'],
                             libraries=['decoding'])]

else:
    ext_modules = [Extension("pydecode.hyper",
                             ["python/pydecode/hyper.cpp"],
                             language='c++',
                             include_dirs=[r'build/debug/src/', "."],
                             library_dirs=[r'build/debug/src/', ""],
                             extra_objects=['build/debug/src/libdecoding.a'],
                             libraries=['decoding'])]
    
setup(
  name = 'pydecode',
  cmdclass = {'build_ext': build_ext},
  packages=['pydecode'],
  package_dir={'pydecode': 'python/pydecode'},
  ext_modules = ext_modules,
  requires=["networkx", "pandas"]
)
