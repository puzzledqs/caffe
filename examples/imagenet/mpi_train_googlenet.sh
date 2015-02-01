#!/usr/bin/env sh

#GOOGLE_LOG_DIR=models/googlenet/log \
mpirun -np 2 ./build/tools/caffe train \
    --solver=examples/imagenet/googlenet_solver.prototxt 2>&1 \
    --snapshot=models/googlenet/bvlc_googlenet_iter_400000.solverstate \
    |tee models/googlenet/log/log_resume_400k2.txt 
#    --gpu=3 \
