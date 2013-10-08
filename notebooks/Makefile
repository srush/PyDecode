all: hypergraphs.py hypergraphs.rst hmm.py hmm.rst tutorial.py tutorial.rst decipher.py decipher.rst

hypergraphs.py: hypergraphs.ipynb
	ipython nbconvert --to python hypergraphs.ipynb

hypergraphs.rst: hypergraphs.ipynb
	ipython nbconvert --to rst hypergraphs.ipynb

hmm.py: hmm.ipynb
	ipython nbconvert --to python hmm.ipynb

hmm.rst: hmm.ipynb
	ipython nbconvert --to rst hmm.ipynb

tutorial.py: tutorial.ipynb
	ipython nbconvert --to python tutorial.ipynb

tutorial.rst: tutorial.ipynb
	ipython nbconvert --to rst tutorial.ipynb

decipher.py: decipher.ipynb
	ipython nbconvert --to python decipher.ipynb

decipher.rst: decipher.ipynb
	ipython nbconvert --to rst decipher.ipynb
