echo 'DOWNLOADING DATA'
wget https://www.dropbox.com/s/eq1iuu4trkjkopt/entity-tracking-data.zip?dl=0 -O entity-tracking-data.zip

echo 'UNZIPPING'
unzip entity-tracking-data.zip

echo 'PREPROCESSING'
python preprocess-lambada.py --data ./entity-tracking-data/lambada/ --glove ./entity-tracking-data/glove.6B.100d.txt --train train.txt --valid valid.txt --test test.txt --std_feats --ent_feats --disc_feats --speaker_feats --out_file lambada.hdf5

echo 'TRAINING'
python train.py -cuda -datafile lambada.hdf5 -save lambada.t7 -dropout 0.2 -bsz 64 -epochs 5 -rnn_size 100 -max_entities 5 -max_mentions 2 -clip 10 -beta1 0.7 -mt_coeff 1.5 -emb_size 100 -std_feats -speaker_feats -maxseqlen 1024 -mt_loss idx-loss -log_interval 1000
