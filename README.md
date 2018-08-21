# Entity Tracking Improves Cloze-style Reading Comprehension #

PyTorch implementation of the paper "Entity Tracking Improves Cloze-style Reading Comprehension". 

## Dependencies
* Python: `h5py`, `numpy`, `nltk`
* Pytorch

## Data
To get started, download the appropriate datasets:
- GLOVE: This code uses 100-d GLOVE embeddings which are available from https://nlp.stanford.edu/projects/glove/. 
- LAMBADA: Our datasets that have been augmented with NER tags and anonymized speaker Ids can be downloaded from: (TODO: fill). The original validation (or development) and test sets were from http://clic.cimec.unitn.it/lambada/. The training set is from: http://people.cs.uchicago.edu/~zeweichu/ (or direct link http://ttic.uchicago.edu/~kgimpel/data/lambada-train-valid.tar.gz).
- CBT Named Entity Dataset: Our datasets that have been augmented with NER tags and anonymized speaker Ids can be downloaded from: (TODO: fill). The original datasets are from https://research.fb.com/downloads/babi/

## Preprocessing
Preprocess the raw train/test/validation text files into .hdf5 files using:
  
    python preprocess.py --data <base_dir> --glove <path_to_glove.6B.100d.txt> --train <train_file> --valid <valid_file> --test <test_file> --std_feats --ent_feats --disc_feats --speaker_feats --out_file <out_hdf5_file>

## Training
A sample training script is given below. For more information, view [`train.py`](https://github.com/harvardnlp/readcomp/blob/master/train.py) for details on parameter settings and descriptions.

    python train.py -cuda -datafile <model_file> -save '<output_save_model_file.t7>' -dropout 0.2 -bsz 64 -epochs 5 -rnn_size 100 -max_entities 5 -max_mentions 2 -clip 10 -beta1 0.7 -mt_coeff 1.5 -emb_size 100 -std_feats -speaker_feats -maxseqlen 1024 -mt_loss idx-loss -log_interval 1000
