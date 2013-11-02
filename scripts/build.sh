python cython_jinja.py
scons build/debug/src/libdecoding.a
sudo rm -fr build/temp.linux-x86_64-2.7/
sudo rm -fr build/lib.linux-x86_64-2.7/
sudo rm -fr python/pydecode/hyper.so
python setup.py clean
python setup.py config
# python setup.py build
python setup.py build_ext --inplace
sudo python setup.py install
# cd notebooks;make all
# cd ../docs;make html
