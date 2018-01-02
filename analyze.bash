#!/bin/bash

trap "kill 0" SIGINT # allow to kill all spawning processes in the same group

# ------------------- Test on various best models --------------------------
python train2.py -datafile lambada.hdf5 -load 'models/best/lamb-reg-best.pt' -cuda -eval_only -analysis
python train2.py -datafile lambada.hdf5 -load 'models/best/lamb-idx-best.pt' -cuda -eval_only -analysis
python train2.py -datafile lambada.hdf5 -load 'models/best/lamb-ant-best.pt' -cuda -eval_only -analysis
python train2.py -datafile cbtest.hdf5 -load 'models/best/cbtne-reg-best.pt' -cuda -eval_only -analysis
python train2.py -datafile cbtest.hdf5 -load 'models/best/cbtne-idx-best.pt' -cuda -eval_only -analysis
python train2.py -datafile cbtest.hdf5 -load 'models/best/cbtne-ant-best.pt' -cuda -eval_only -analysis

# ------------------- LAMBADA --------------------------

# python preprocess-lambada.py --data ~/data/lambada/lambada-sam/original/ --glove ~/data/glove/glove.6B.100d.txt --train train.txt --valid lambadev_replace.txt --test test.txt --std_feats --ent_feats --disc_feats --speaker_feats --out_file lambada.hdf5

# python train2.py -cuda -seed 13 -dropout 0.2 -bsz 64 -epochs 1 -rnn_size 256 -max_entities 5 -max_mentions 5 -clip 10 -datafile lambada.hdf5 -emb_size 128 -std_feats -speaker_feats -maxseqlen 1028 -mt_loss idx-loss -log_interval 1000 -save 'models/lambada.t7'

# python train2.py -load 'models/lambada.t7' -datafile lambada.hdf5 -std_feats -speaker_feats -emb_size 128 -rnn_size 256 -log_interval 1000 -bsz 64 -cuda -clip 10 -maxseqlen 1028 -mt_loss idx-loss -max_entities 5 -max_mentions 5 -eval_only -analysis

# ------------------- CBT --------------------------

# python preprocess-lambada.py --data ~/data/cbt/ --glove ~/data/glove/glove.6B.100d.txt --train NE_train.lq.replace --valid NE_valid.lq.replace --test NE_test.lq.replace --std_feats --ent_feats --disc_feats --speaker_feats --cbt_mode --answer_identifier xxxxx --out_file cbt.hdf5

# python train2.py -datafile cbt.hdf5 -dropout 0.2 -std_feats -speaker_feats -emb_size 256 -rnn_size 256 -log_interval 1000 -bsz 64 -add_inp -cuda -clip 10 -use_choices -maxseqlen 1500 -use_qidx -mt_loss "idx-loss" -max_entities 3 -max_mentions 3 -transform_for_ants -mt_step_mode before -beta1 0.9 -mt_coeff 0.5 -lr 0.001 -epochs 4 -save 'models/cbt-ne.t7'

# python train2.py -load 'models/cbt-ne.t7' -datafile cbt.hdf5 -dropout 0.2 -std_feats -speaker_feats -emb_size 256 -rnn_size 256 -log_interval 1000 -bsz 64 -add_inp -cuda -clip 10 -use_choices -maxseqlen 1500 -use_qidx -mt_loss idx-loss -max_entities 3 -max_mentions 3 -transform_for_ants -mt_step_mode before -beta1 0.9 -mt_coeff 0.5 -lr 0.001 -epochs 4 -eval_only -analysis

echo "Done"
