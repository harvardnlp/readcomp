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

end_words  = { "?", "??", "???", "!", "!!", "!!!", ".", "?!", "!?" }
quote_words = { "'", "\"" }

def load_punc(punc_file):
  punctuations = {}
  with open(punc_file, 'r') as f:
    for line in f:
      punc = line.strip()
      if punc:
        punctuations[punc] = 1
  return punctuations


def aggregate(in_path, out_file, punctuations, out_entity_vocab, separator, num_lines_to_process, debug):
  print 'Processing path {}'.format(in_path)

  num_processed_lines = 0
  os.chdir(in_path)
  with codecs.open(out_file, 'w', encoding='utf8') as outf:
    for file in glob.glob("*.question"):
      with codecs.open(file, 'r', encoding='utf8') as f:
        url = f.readline().strip()
        f.readline()
        context = f.readline().strip()
        f.readline()
        query = f.readline().strip()
        f.readline()
        answer = f.readline().strip()
        f.readline()

        if len(separator.strip()) == 0:
          raise Exception('A separator token must be specified to distinguish context, query, and answer')

        doc = context + ' ' + separator + ' ' + query + ' ' + separator + ' ' + answer + '\n'
        outf.write(doc)

        doc_words = doc.split()
        for w in doc_words:
          if w[0] == '@' and len(w) > 1 and w not in out_entity_vocab:
            out_entity_vocab[w] = 1

        num_processed_lines += 1
        if num_lines_to_process > 0 and num_processed_lines >= num_lines_to_process:
          break
        
        if debug:
          mapping = {}
          for line in f:
            parts = line.split(':')
            mapping[parts[0].strip()] = parts[1].strip()
          d = doc.split(' ')
          c = []
          for token in d:
            if token in mapping:
              c.append(mapping[token])
            else:
              c.append(token)
          print(' '.join(c))
          quit = raw_input('Continue? ("q" to quit): ')
          if quit == 'q' or quit == 'quit':
            break


def main(arguments):
  global args
  parser = argparse.ArgumentParser(
      description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter) 
  parser.add_argument('--data', type=str, default='/home/lhoang/data/cnn/questions/',
                      help='location of the CNN question corpus')
  parser.add_argument('--train', type=str, default='training',
                      help='relative folder of training data')
  parser.add_argument('--valid', type=str, default='validation',
                      help='relative folder of validation data')
  parser.add_argument('--test', type=str, default='test',
                      help='relative folder of testing data')
  parser.add_argument('--punctuations', type=str, default='punctuations.txt',
                      help='relative location of punctuation file')
  parser.add_argument('--out_train_file', type=str, default='train.txt',
                      help='relative location of output training file')
  parser.add_argument('--out_valid_file', type=str, default='valid.txt',
                      help='relative location of output validation file')
  parser.add_argument('--out_test_file', type=str, default='test.txt',
                      help='relative location of output testing file')
  parser.add_argument('--out_control_file', type=str, default='control.txt',
                      help='relative location of output control file')
  parser.add_argument('--out_entity_vocab_file', type=str, default='entity_vocab.txt',
                      help='relative location of output entity vocab file')
  parser.add_argument('--context_query_separator', type=str, default='$$$',
                      help='separator token between context, query and answer')
  parser.add_argument('--tiny', action='store_true',
                      help='whether to generate tiny files')
  parser.add_argument('--debug', action='store_true',
                      help='print output for manual debugging')

  args = parser.parse_args(arguments)

  punctuations = load_punc(args.data + args.punctuations)

  out_entity_vocab = {}
  aggregate(args.data + args.train, args.data + args.out_train_file,   punctuations, out_entity_vocab, args.context_query_separator, 1000 if args.tiny else -1, args.debug)
  aggregate(args.data + args.valid, args.data + args.out_valid_file,   punctuations, out_entity_vocab, args.context_query_separator, 100 if args.tiny else -1, args.debug)
  # just a dummy control file to fit the syntax of later processing code
  aggregate(args.data + args.valid, args.data + args.out_control_file, punctuations, out_entity_vocab, args.context_query_separator, 100 if args.tiny else -1, args.debug)
  aggregate(args.data + args.test,  args.data + args.out_test_file,    punctuations, out_entity_vocab, args.context_query_separator, 100 if args.tiny else -1, args.debug)

  with codecs.open(args.data + args.out_entity_vocab_file, 'w', encoding='utf8') as vf:
    for w in out_entity_vocab:
      vf.write(w + '\n')


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))