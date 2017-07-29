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
import glob, os
import csv
import re

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def get_vocab(folder, word_to_idx = {}):
    idx = len(word_to_idx) + 1
    owd = os.getcwd()
    os.chdir(folder)
    for infile in glob.glob("*.txt"):
        with open(infile) as inf:
            for line in inf:
                parts = line.lower().replace('.','').strip().split('?')
                tokens = parts[0].split(' ')
                for i in np.arange(1, len(tokens)):
                    if tokens[i] not in word_to_idx and not is_number(tokens[i]):
                        word_to_idx[tokens[i]] = idx
                        idx += 1
                if len(parts) > 1:
                    answer = parts[1].strip().split('\t')[0]
                    if answer not in word_to_idx:
                        word_to_idx[answer.lower()] = idx
                        idx += 1
    os.chdir(owd)
    return word_to_idx

def process(file):
  contexts = []

  current_story = ''
  with open(file) as inf:
    for line in inf:
      line_info = re.match('([0-9].*?)\ (.*)', line)
      line_no = int(line_info.group(1)) # line number
      line_data = line_info.group(2) # rest of line

      parts = line_data.lower().split('?')

      if line_no == 1:
        current_story = ''

      if len(parts) > 1:
        answer = parts[1].strip().split('\t')[0]
        contexts.append(current_story + parts[0].strip() + ' ' + answer.strip())
      else:
        current_story = current_story + parts[0].strip().replace('.', ' . ')

  return contexts


def process_files(folder, out_train_file, out_test_file, out_valid_file, out_control_file):
  with open(out_train_file, 'w') as trainf:
    with open(out_test_file, 'w') as testf:
      with open(out_valid_file, 'w') as validf:
        with open(out_control_file, 'w') as controlf:
          owd = os.getcwd()
          os.chdir(folder)
          
          for infile in glob.glob("qa*.txt"):
            file_info = re.match('qa(.*)_(.*)_(.*).txt', infile)
            task_data_type = file_info.group(3) # train or test

            # process data
            contexts = process(infile)
            if task_data_type == 'train':
              for c in contexts:
                trainf.write(c + "\n")
            else:
              for c in contexts:
                rand = np.random.rand()
                if rand < (1.0/3):
                  testf.write(c + "\n")
                elif rand < (2.0/3):
                  validf.write(c + "\n")
                else:
                  controlf.write(c + "\n")

          os.chdir(owd)


def main(arguments):
    global args
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-trainfile', help="name of the training file to create", type=str,default='train.txt',required=False)
    parser.add_argument('-testfile', help="name of the test file to create", type=str,default='test.txt',required=False)
    parser.add_argument('-validfile', help="name of the validation file to create", type=str,default='valid.txt',required=False)
    parser.add_argument('-controlfile', help="name of the control file to create", type=str,default='control.txt',required=False)
    parser.add_argument('-vocabfile', help="name of the vocab file to create", type=str,default='vocab.txt',required=False)
    parser.add_argument('-dir', help="data directory",
                        type=str,default='/mnt/c/Users/lhoang/Dropbox/Personal/Work/lambada-dataset/babi_data/en/',required=False)
    args = parser.parse_args(arguments)

    word_to_idx = get_vocab(args.dir)
    with open(args.vocabfile, 'w') as vocabfile:
      for word in word_to_idx:
        vocabfile.write(word + "\n")

    process_files(args.dir, args.trainfile, args.testfile, args.validfile, args.controlfile)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
