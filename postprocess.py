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
import time
import seaborn as sns
import matplotlib.pyplot as plt
import datamodel
from pylab import savefig

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
  parser.add_argument('--out_vocab_file_prefix', type=str, default='lambada-asr',
                      help='file name prefix of vocab files')
  parser.add_argument('--verbose_level', type=int, default=2,
                      help='level of verbosity, ranging from 0 to 2, default = 2')

  args = parser.parse_args(arguments)

  hfont = { 'fontname':'serif' }
  sns.set(font_scale=2,font='serif')
  plt.figure(figsize=(54, 24), dpi=100)

  corpus = datamodel.Corpus(args.verbose_level, None, None, None, None, None, None, None, None)
  corpus.load_vocab(args.out_vocab_file_prefix)
  idx2word = corpus.dictionary.idx2word

  total = 0
  correct = 0
  for filename in glob.glob(args.model_dump_pattern):
    with h5py.File(filename, "r") as f:
      inputs      = np.array(f['inputs'],      dtype=int)
      targets     = np.array(f['targets'],     dtype=int)
      outputs     = np.array(f['outputs'],     dtype=float)
      predictions = np.array(f['predictions'], dtype=int)
      answers     = np.array(f['answers'],     dtype=int)

      batch_size, max_len = outputs.shape
      answer_index = answers > 0
      total += np.sum(answer_index)
      correct += np.sum(predictions[answer_index] == answers[answer_index])

      dim1 = int(np.floor(np.sqrt(max_len)))
      dim2 = int(np.ceil(float(max_len) / dim1))

      ip = np.concatenate((inputs.T, np.zeros((batch_size, dim1 * dim2 - max_len))), axis = 1)
      op = np.concatenate((outputs,  np.zeros((batch_size, dim1 * dim2 - max_len))), axis = 1)

      txt = []
      for i in range(ip.shape[0]):
        r = []
        for j in range(ip.shape[1]):
          word = idx2word[int(ip[i,j])] if ip[i,j] != 0 else ''
          r.append(u'{}\n{:.4f}'.format(word, op[i,j]))
        txt.append(np.array(r))
      labels = np.array(txt)

      for b in range(batch_size):
        ax = plt.axes()
        sns.heatmap(op[b].reshape((dim1,dim2)), annot=labels[b].reshape((dim1,dim2)), fmt='', cmap='Blues', ax = ax)
        correct_indicator = r"$\bf{CORRECT}$" if answers[b] == predictions[b] else "Correct"
        target_sentence = ' '.join([idx2word[idx] if idx != 0 else '' for idx in targets[:,b]]).strip()
        ax.set_title('{} = {}, Prediction = {}\nTarget = {}'.format(correct_indicator, idx2word[answers[b]], idx2word[predictions[b]], target_sentence))
        # plt.show()
        savefig(filename + '.ex{}.png'.format(str(b).zfill(3)), bbox_inches='tight', dpi = 100)
        plt.clf()

  print 'Correct = {}, Total = {}, % = {:.4f}%'.format(correct, total, float(correct) * 100 / total)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
