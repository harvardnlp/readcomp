#!/usr/bin/env python

import numpy as np
import h5py
import argparse
import sys
import re
import codecs
import copy
import operator
import json
import glob, os, fnmatch
import csv
import re
import nltk
import time
import datamodel

args = {}

def main(arguments):
  global args
  parser = argparse.ArgumentParser(
      description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter) 
  parser.add_argument('--data', type=str, default='~/Dropbox/Personal/Work/lambada-dataset/lambada-train-valid/original/',
                      help='location of the data corpus')
  parser.add_argument('--glove', type=str, default='~/data/glove/glove.6B.100d.txt', # e.g. data/glove.6B.100d.txt
                      help='absolute path of the glove embedding file')
  parser.add_argument('--glove_size', type=int, default=-1,
                      help='size of the vocab to build from glove embeddings')
  parser.add_argument('--model_dump_pattern', type=str, default='*.t7.*.dump',
                      help='file pattern of model dumps for analysis')
  parser.add_argument('--train', type=str, default='train.txt',
                      help='relative location (file or folder) of training data')
  parser.add_argument('--valid', type=str, default='valid.txt',
                      help='relative location (file or folder) of validation data')
  parser.add_argument('--test', type=str, default='test.txt',
                      help='relative location (file or folder) of testing data')
  parser.add_argument('--control', type=str, default='control.txt',
                      help='relative location (file or folder) of control data')
  parser.add_argument('--vocab', type=str, default='vocab.txt',
                      help='relative location of vocab file')
  parser.add_argument('--extra_vocab', type=str, default=None,
                      help='relative location of extra vocab file to load into dictionary')
  parser.add_argument('--punctuations', type=str, default='punctuations.txt',
                      help='relative location of punctuation file')
  parser.add_argument('--stopwords', type=str, default='mctest-stopwords.txt',
                      help='relative location of stop-words file')
  args = parser.parse_args(arguments)
  # get embeddings
  # word_to_idx, suffix_to_idx, prefix_to_idx, embeddings = get_vocab_embedding(args.vocabsize)

  total = 0
  correct = 0
  for filename in glob.glob(args.model_dump_pattern):
    with h5py.File(filename, "r") as f:
      inputs      = np.array(f['inputs'])
      outputs     = np.array(f['outputs'])
      predictions = np.array(f['predictions'])
      answers     = np.array(f['answers'])

      answer_index = answers > 0
      total += np.sum(answer_index)
      correct += np.sum(predictions[answer_index] == answers[answer_index])

  print 'Correct = {}, Total = {}, % = {:.4f}%'.format(correct, total, float(correct) * 100 / total)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
