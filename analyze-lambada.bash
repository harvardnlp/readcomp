#!/bin/bash

trap "kill 0" SIGINT # allow to kill all spawning processes in the same group

python preprocess-lambada.py --data ~/data/lambada/lambada-sam/original/ --glove ~/data/glove/glove.6B.100d.txt --train train.txt --valid lambadev_replace.txt --test test.txt --std_feats --ent_feats --disc_feats --speaker_feats --out_file lambada.hdf5

python train2.py -datafile lambada.hdf5 -dropout 0.1 -std_feats -speaker_feats -emb_size 128 -rnn_size 128 -log_interval 1000 -bsz 64 -add_inp -cuda -clip 10 -maxseqlen 1028 -mt_loss idx-loss -max_entities 2 -max_mentions 2

echo "Done"
