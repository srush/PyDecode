from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension("decoding_ext",
                         ["python/decoding/decoding_ext.pyx"],
                         language='c++',
                         include_dirs=[r'build/debug/src/'],
                         library_dirs=[r'build/debug/src/', ""],
                         extra_objects=['build/debug/src/libdecoding.a'],
                         libraries=['decoding'])]

setup(
  name = 'decoding',
  cmdclass = {'build_ext': build_ext},
  packages=['decoding', 'decoding.interfaces'],
  package_dir={'decoding': 'python/decoding'},
  ext_modules = ext_modules,
)
