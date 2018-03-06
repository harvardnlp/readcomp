# Improved Entity Tracking for Cloze-style Reading Comprehension #

PyTorch implementation of the paper "Improved Entity Tracking for Cloze-style Reading Comprehension". 

## Dependencies
* Python: `h5py`, `numpy`, `nltk`
* Pytorch

## Data
To get started, download the appropriate datasets:
- GLOVE: This code uses 100-d GLOVE embeddings which are available from https://nlp.stanford.edu/projects/glove/. 
- LAMBADA: The validation (or development) and test sets are from http://clic.cimec.unitn.it/lambada/. The training set is obtained from: http://people.cs.uchicago.edu/~zeweichu/ (or direct link http://ttic.uchicago.edu/~kgimpel/data/lambada-train-valid.tar.gz).
- CBT Named Entity Dataset: Download from https://research.fb.com/downloads/babi/

## Preprocessing
1. First, augment the datasets with NER tags and Speaker Ids using (TODO: fill)
2. Then, anonymize the entities with the following commands: `python anonymize.py --input_file <input-file> --output_file <output-file>`, where input file is the output of step (1).
3. Finally, preprocess the raw train/test/validation text files into .hdf5 files using:
  
    `python preprocess-lambada.py --data ~/data/lambada/lambada-sam/original/ --glove ~/data/glove/glove.6B.100d.txt --train train.txt --valid lambadev_replace.txt --test test.txt --std_feats --ent_feats --disc_feats --speaker_feats --out_file /mnt/models/lambada.hdf5`

## Training

    python train.py -cuda -dropout 0.1 -bsz 64 -epochs 3 -rnn_size 256 -max_entities 2 -max_mentions 2 -clip 10 -datafile lambada.hdf5 -emb_size 128 -std_feats -speaker_feats -maxseqlen 1028 -mt_loss idx-loss -log_interval 1000 -save 'models/lambada.t7'

For more information, view `train.py` for details on parameter settings and descriptions.
