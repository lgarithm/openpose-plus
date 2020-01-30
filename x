#!/bin/sh
set -e

# git clean -fdx
# rm -fr 3rdparty

./configure
make
./bin/example-main
