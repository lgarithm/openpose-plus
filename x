#!/bin/sh
set -e

# git clean -fdx
# rm -fr 3rdparty

config_flags() {
    echo --with-blas=$HOME/local/openblas
}

./configure $(config_flags)
make
./bin/example-main
