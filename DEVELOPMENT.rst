To contribute to development please email srush at csail mit edu or send a pull request.



Build Commands
===============

Run to build a debug, profile and optimized version of the libdecode library ::

   > scons


Run to build the C++ and python documentation. The index page is in "docs/build/index.html" ::

   > scons docs


Builds and runs the C++ tests. (Requires gtest). ::

   > scons test

Runs the python tests. (Requires py.test and nosetests). ::

   > scons pytest

Runs a C++ lint check. Uses the Google C++ style guide. ::

   > bash script/check_syntax.sh

Runs a python lint check. (Requires pep8) ::

   > bash script/check_python_syntax.sh

Builds a clean version of the library and Cython code. ::

   > bash script/build.sh

Build (and install) the python extension.

   > python setup.py config
   > python setup.py build
   > python setup.py build_ext --inplace
   > sudo python setup.py install


Documentation
=============

C++ indoc documentation is in Doxygen format.

http://www.stack.nl/~dimitri/doxygen/manual/docblocks.html

Python indoc documentation is in numpydoc format.

https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt


Directories
============

src/ - C++ library code.

python/ - Python library code.

notebooks/ - IPython notebooks (Included as documentation).

docs/ - Sphinx documentation.

scripts/ - misc. scripts

writing/ - misc. associated documents.