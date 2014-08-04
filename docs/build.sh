cd notebooks/
ipython nbconvert *.ipynb --to rst
cd doc/
ipython nbconvert *.ipynb --to rst
cd ../../
make html
