#!/bin/bash

trap "kill 0" SIGINT # allow to kill all spawning processes in the same group

cdnlp
python preprocess-cnn.py --data ~/data/cnn/questions/ --tiny
python preprocess-lambada.py --data ~/data/cnn/questions/ --glove ~/data/glove.6B.100d.txt --extra_vocab entity_vocab.txt --out_file cnn.hdf5
th train.lua --cuda --device 1 --randomseed 7 --model asr --batchsize 64 --postsize 80 --gcepoch 1000 --maxseqlen 10000 >> $OUTFILE

echo "Done"
