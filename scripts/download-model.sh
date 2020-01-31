#!/bin/sh
set -e

# 0962749be446b10fd2843efcaab1cc5f3fd92273 hao28-pose600000.npz
# 72f05f9e906f3dc08d8f78cb09f56985994a8d95 hao28-pose600000.idx.tar

MODEL_DIR=$HOME/var/models/openpose

mkdir -p $MODEL_DIR
cd $MODEL_DIR

curl -vLOJ https://github.com/tensorlayer/pretrained-models/raw/master/models/openpose-plus/hao28-pose600000.npz
