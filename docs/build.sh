cd notebooks/doc/
ipython nbconvert *.ipynb --to rst
cd ../../
make html
