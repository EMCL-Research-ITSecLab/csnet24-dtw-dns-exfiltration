#!/bin/bash

source .venv/bin/activate
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda


python train.py lstm > lstm.log
python train.py hivecotev > hivecotev.log
python train.py knn-euclidean > knn-euclidean.log
python train.py knn-dtw > knn-dtw.log
python train.py knn-dtw-sakoe > knn-dtw-sakoe.log
python train.py knn-dtw-itakura > knn-dtw-itakura.log
