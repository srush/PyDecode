Build Commands
----------------

> scons

Run to build a debug, profile and optimized version of the libdecode library.

> scons docs

Run to build the C++ and python documentation. The index page is in "docs/build/index.html"

> scons test

Builds and runs the C++ tests. (Requires gtest).

> scons pytest

Runs the python tests. (Requires py.test and nosetests).

> bash script/check_syntax.sh

Runs a C++ lint check. Uses the Google C++ style guide.

> bash script/check_python_syntax.sh

Runs a python lint check. (Requires pep8)

> bash script/build.sh

Builds a clean version of the library and Cython code.

> python setup.py config
> python setup.py build
> python setup.py build_ext --inplace
> sudo python setup.py install

Build (and install) the python extension.


Documentation
--------------

C++ indoc documentation is in Doxygen format.

http://www.stack.nl/~dimitri/doxygen/manual/docblocks.html

Python indoc documentation is in numpydoc format.

https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt


Directories
-----------

src/ - C++ library code.
python/ - Python library code.
notebooks/ - IPython notebooks (Included as documentation).
docs/ - Sphinx documentation.
scripts/ - misc. scripts
writing/ - misc. associated documents.
