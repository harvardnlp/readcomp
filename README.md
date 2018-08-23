# Entity Tracking Improves Cloze-style Reading Comprehension #

PyTorch implementation of the paper "Entity Tracking Improves Cloze-style Reading Comprehension". 

## Tested environment:
* Python 2.7.14: `h5py`, `numpy`, `nltk`
* Pytorch 0.3.1 (**important, later versions may not work**): https://pytorch.org/previous-versions/
* Cuda 8.0

## TLDR:
After setting up the environment, one can directly run [`train.bash`](https://github.com/harvardnlp/readcomp/blob/master/train.bash) to:
1. Download data
2. Preprocess datasets into hdf5 format
3. Train a model on the Lambada dataset

## Data
The relevant datasets can be downloaded from [here](https://www.dropbox.com/s/eq1iuu4trkjkopt/entity-tracking-data.zip?dl=0), which include:
- GLOVE: 100-d GLOVE embeddings which are originally available from https://nlp.stanford.edu/projects/glove/. 
- LAMBADA: Our versions that have been augmented with NER Tags and Anonymized Speaker Ids. The original validation (or development) and test sets were from http://clic.cimec.unitn.it/lambada/. The training set is from: http://people.cs.uchicago.edu/~zeweichu/ (or direct link http://ttic.uchicago.edu/~kgimpel/data/lambada-train-valid.tar.gz).
- CBT Named Entity Dataset: Our versions that have been augmented with NER Tags and Anonymized Speaker Ids can be downloaded from. The original datasets are from https://research.fb.com/downloads/babi/

## Preprocessing
Preprocess the raw train/test/validation text files into .hdf5 files using:
  
    python preprocess.py --data <base_dir> --glove <path_to_glove.6B.100d.txt> --train <train_file> --valid <valid_file> --test <test_file> --std_feats --ent_feats --disc_feats --speaker_feats --out_file <out_hdf5_file>

## Training
A sample training script is given below. For more information, view [`train.py`](https://github.com/harvardnlp/readcomp/blob/master/train.py) for details on parameter settings and descriptions.

    python train.py -cuda -datafile <model_file> -save <output_save_model_file.t7> -dropout 0.2 -bsz 64 -epochs 5 -rnn_size 100 -max_entities 5 -max_mentions 2 -clip 10 -beta1 0.7 -mt_coeff 1.5 -emb_size 100 -std_feats -speaker_feats -maxseqlen 1024 -mt_loss idx-loss -log_interval 1000
