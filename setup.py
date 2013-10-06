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
cmdclass = {}

if use_cython:
    ext_modules = [Extension("pydecode.hyper",
                             ["python/pydecode/hyper.pyx"],
                             language='c++',
                             include_dirs=[r'build/opt/src/', "."],
                             library_dirs=[r'build/opt/src/', ""],
                             extra_objects=['build/opt/src/libdecoding.a'],
                             libraries=['decoding'])]
    cmdclass = {'build_ext': build_ext}

else:
    ext_modules = [Extension("pydecode.hyper",
                             ["python/pydecode/hyper.cpp"],
                             language='c++',
                             include_dirs=[r'build/opt/src/', "."],
                             library_dirs=[r'build/opt/src/', ""],
                             extra_objects=['build/opt/src/libdecoding.a'],
                             libraries=['decoding'])]

setup(
  name = 'pydecode',
  cmdclass = cmdclass,
  packages=['pydecode'],
  package_dir={'pydecode': 'python/pydecode'},
  ext_modules = ext_modules,
  requires=["networkx", "pandas"]
)
