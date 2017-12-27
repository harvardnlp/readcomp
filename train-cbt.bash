# python preprocess-lambada.py --data ~/data/cbt/ --glove ~/data/glove/glove.6B.100d.txt --train NE_train.lq.replace --valid NE_valid.lq.replace --test NE_test.lq.replace --std_feats --ent_feats --disc_feats --speaker_feats --cbt_mode --answer_identifier xxxxx --out_file cbt.hdf5

# th nutrain.lua --cuda --randomseed 13 --maxepoch 7 --postsize 128 --id CE35choicemulti --datafile NE_cbt.hdf5 --nersize 128 --sentsize 80 --speesize 80 --entity 2 --entitysize 2 --std_feats --ent_feats --disc_feats --speaker_feats --use_choices --lr 0.001 --maxseqlen 1500 --dropout 0.2

python train2.py -datafile cbt.hdf5 -dropout 0.2 -std_feats -speaker_feats -emb_size 256 -rnn_size 256 -log_interval 1000 -bsz 64 -add_inp -cuda -clip 10 -use_choices -maxseqlen 1500 -use_qidx -mt_loss "idx-loss" -max_entities 3 -max_mentions 3 -transform_for_ants -mt_step_mode before -beta1 0.9 -mt_coeff 0.5 -lr 0.001 -epochs 4 -save 'models/cbt-ne.t7'
