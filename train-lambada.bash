#!/bin/bash

trap "kill 0" SIGINT # allow to kill all spawning processes in the same group

python preprocess-lambada.py --data ~/data/lambada/lambada-train-valid/original/ --glove ~/data/glove/glove.6B.100d.txt --out_file lambada-asr.hdf5
th train.lua --cuda --device 1 --randomseed 7 --model asr --batchsize 64 --postsize 80 --gcepoch 1000 --maxseqlen 1024 --datafile lambada-asr.hdf5

echo "Done"
