# Improved Entity Tracking for Cloze-style Reading Comprehension #

PyTorch implementation of the paper "Improved Entity Tracking for Cloze-style Reading Comprehension". To get started, download the appropriate datasets:
- For LAMBADA, the validation (or development) and test sets are from http://clic.cimec.unitn.it/lambada/. The training set is obtained from: http://people.cs.uchicago.edu/~zeweichu/ (or direct link http://ttic.uchicago.edu/~kgimpel/data/lambada-train-valid.tar.gz).
- CBT: (TODO: fill)

This code also uses 100-d GLOVE embeddings, which is from https://nlp.stanford.edu/projects/glove/. 

Preprocessing:
1. First, augment the datasets with NER tags and Speaker Ids using (TODO: fill)
2. Then, anonymize the entities with the following commands: `python anonymize.py --input_file <input-file> --output_file <output-file>`, where input file is the output of step (1).
3. Finally, preprocess the raw train/test/validation text files into .hdf5 files using:
  
    `python preprocess-lambada.py --data ~/data/lambada/lambada-sam/original/ --glove ~/data/glove/glove.6B.100d.txt --train train.txt --valid lambadev_replace.txt --test test.txt --std_feats --ent_feats --disc_feats --speaker_feats --out_file /mnt/models/lambada.hdf5`
