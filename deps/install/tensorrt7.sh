#!/bin/sh
set -e

filename=nv-tensorrt-repo-ubuntu1804-cuda10.2-trt7.0.0.11-ga-20191216_1-1_amd64.deb

cd $(dirname $0)/..

hash=$(cat sha1sum.txt | grep $filename)

find_deb() {
    local folder=$1
    if [ $(ls $folder | grep $filename | wc -l) -eq 1 ]; then
        echo $folder
        cd $folder
        echo $hash | sha1sum -c -
        sudo dpkg -i $filename
    fi
}


# find_deb $HOME


check_install(){
    local suffix="_1-1_amd64.deb"
    local pkg_name=${filename%$suffix}
    if [ $(apt list --installed | grep $pkg_name | wc -l) -gt 0 ]; then
        echo OK
    fi
}

check_install

