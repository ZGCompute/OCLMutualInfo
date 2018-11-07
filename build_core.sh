#!/bin/bash

# Abort install if any errors occur
set -o errexit

export BASEDIR=~/Desktop/clDev/OCLMutualInfo/

# Compile ocl.* module
cd $BASEDIR
if [ ! -d "build" ]; then
    mkdir build
fi

make
sudo make install


