#!/bin/sh
set -e

cd $(dirname $0)

latest() {
    local name=$1
    local tag=$2
    local repo=$3
    # local url=https://github.com/$repo
    local url=git@github.com:$repo.git

    if [ ! -d $repo ]; then
        git clone $url github.com/$repo
    fi
    pushd github.com/$repo
    # git checkout $tag
    git pull
    # local commit=$(git ls-remote $url $tag | awk '{print $1}')
    # echo "$commit $tag" | tee $name
    popd
}

latest stdtracer master lgarithm/stdtracer
latest stdtensor dev-0.6.0 lgarithm/stdtensor
latest stdnn-ops grad-kernel lgarithm/stdnn-ops
