#!/bin/sh
set -e

now() { date +%s; }

measure() {
    local begin=$(now)
    $@
    local end=$(now)
    local duration=$((end - begin))
    echo "$@ took ${duration}s"
}

# git clean -fdx
# rm -fr 3rdparty

config_flags() {
    echo --with-blas=$HOME/local/openblas
}

rebuild() {
    ./configure $(config_flags)
    make
}

measure rebuild
measure ./bin/example-main
