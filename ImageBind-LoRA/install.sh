#!/usr/bin/env sh
# Set the compiler to gcc-9 and g++-9
export CC=/usr/bin/gcc-9
export CXX=/usr/bin/g++-9

HOME=`pwd`

# Chamfer Distance
cd $HOME/extensions/chamfer_dist
python setup.py install --user

# EMD
cd $HOME/extensions/emd
python setup.py install --user
