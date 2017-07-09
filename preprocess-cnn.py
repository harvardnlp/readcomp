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


def aggregate(in_path, out_file, debug):
  print 'Processing path {}'.format(in_path)
  os.chdir(in_path)
  for file in glob.glob("*.question"):
    with codecs.open(file, 'r', encoding='utf8') as f:
      with codecs.open(out_file, 'w', encoding='utf8') as outf:
        url = f.readline().strip()
        f.readline()
        context = f.readline().strip()
        f.readline()
        query = f.readline().strip()
        f.readline()
        answer = f.readline().strip()
        f.readline()

        doc = context + ' . ' + query + ' ' + answer + '\n'
        
        mapping = {}
        for line in f:
          parts = line.split(':')
          mapping[parts[0].strip()] = parts[1].strip()

        if debug:
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

        outf.write(doc)


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
  parser.add_argument('--out_train_file', type=str, default='train.txt',
                      help='relative location of output training file')
  parser.add_argument('--out_valid_file', type=str, default='valid.txt',
                      help='relative location of output validation file')
  parser.add_argument('--out_test_file', type=str, default='test.txt',
                      help='relative location of output testing file')
  parser.add_argument('--debug', action='store_true',
                      help='print output for manual debugging')

  args = parser.parse_args(arguments)

  aggregate(args.data + args.train, args.data + args.out_train_file, args.debug)
  aggregate(args.data + args.valid, args.data + args.out_valid_file, args.debug)
  aggregate(args.data + args.test,  args.data + args.out_test_file,  args.debug)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))