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

args = {}


def print_msg(message, verbose_level):
  if args.verbose_level >= verbose_level:
    print message


def rindex(mylist, myvalue):
  try:
    index = len(mylist) - mylist[::-1].index(myvalue) - 1
  except:
    index = -1
  return index


class Dictionary(object):
  def __init__(self):
    self.word2idx = {}
    self.word2count = {}
    self.idx2word = []

  def add_word(self, word):
    if word not in self.word2idx:
      self.idx2word.append(word)
      self.word2idx[word] = len(self.idx2word) - 1
      self.word2count[word] = 0 # set to 0 since a word in vocab may not appear in training data
    return self.word2idx[word]

  def update_count(self, word):
    self.word2count[word] = self.word2count[word] + 1 # the word must be part of the vocab

  def __len__(self):
    return len(self.idx2word)

  def write_to_file(self, file):
    with open(file, 'w') as outf:
      for i in range(len(self.idx2word)):
        word = self.idx2word[i]
        count = self.word2count[word]
        outf.write('{}\t{}\t{}\n'.format(i,word,count))


class Corpus(object):
  def __init__(self, path, vocab):
    self.dictify(os.path.join(path, vocab))


  def dictify(self, file):
    self.dictionary = Dictionary()
    self.dictionary.add_word('<sep>') # map to 0 for masked rnn
    self.dictionary.add_word('<unk>')
    with open(file, 'r') as f:
      for line in f:
        words = line.split()
        for word in words:
          self.dictionary.add_word(word)
    print_msg('Vocab size = {}'.format(len(self.dictionary)), verbose_level = 1)


  def load(self, path, train, valid, test, control, max_context_length):
    self.train   = self.tokenize(os.path.join(path, train), max_context_length, training = True)
    self.valid   = self.tokenize(os.path.join(path, valid), 0, training  = False)
    self.test    = self.tokenize(os.path.join(path, test), 0, training = False)
    self.control = self.tokenize(os.path.join(path, control), 0, training = False)

    print_msg('\nTraining Data Statistics:\n', verbose_level = 1)
    train_context_length = np.array(self.train['context_length'])
    train_context_length = train_context_length[train_context_length > 0]
    print_msg('Context Length: max = {}, min = {}, average = {}, std = {}'.format(
      np.max(train_context_length), np.min(train_context_length), np.mean(train_context_length), np.std(train_context_length)), verbose_level = 1)

    train_target_length = np.array(self.train['target_length'])
    train_target_length = train_target_length[train_target_length > 0]
    print_msg('Target Length: max = {}, min = {}, average = {}, std = {}'.format(
      np.max(train_target_length), np.min(train_target_length), np.mean(train_target_length), np.std(train_target_length)), verbose_level = 1)


  def tokenize(self, path, max_context_length, training):
    """
    Tokenizes all data found recursively in a path (supports single file or folder).
    :param max_context_length: Sentences will be concatenated up to this length.
        If a single sentence is longer it is unchanged.
        Set to 0 to treat each sentence separately. 
    """
    assert os.path.exists(path)
    
    data = { 
      'data': [], # token ids for each word in the corpus 
      'offsets': [], # offset locations for each line in the final 1-d data array 
      'context_length': [], # count of words in the context (excluding target)
      'target_length': [] # count of words in the target
    }

    if os.path.isdir(path):
      num_files = 0
      for root, dir_names, file_names in os.walk(path):
        for file in fnmatch.filter(file_names, '*.txt'):
          num_files += 1

      num_processed_files = 0
      for root, dir_names, file_names in os.walk(path):
        for file in fnmatch.filter(file_names, '*.txt'):
          if training:
            self.tokenize_train_file(os.path.join(root, file), max_context_length, data)
          else:
            self.tokenize_test_file(os.path.join(root, file), max_context_length, data)
          num_processed_files += 1
          print_msg('Progress = {:2.2f}%'.format(num_processed_files * 100.0 / num_files), verbose_level = 1)
    else:
      if training:
        self.tokenize_train_file(path, max_context_length, data)
      else:
        self.tokenize_test_file(path, max_context_length, data)
    return data


  # update the ids, offsets, word counts, line counts
  def tokenize_train_file(self, file, max_context_length, data):
    # Tokenize file content
    target_sentence = []
    to_be_processed = [] # all sentences to be processed
    num_lines_in_file = 0
    num_lines_processed = 0
    with open(file, 'r') as f:
      running_context_length = 0
      for line in f:
        num_lines_in_file += 1
        words = line.split()
        to_be_processed.append(words)

        data['offsets'].append(len(data['data']) + 1)

        # found enough data 
        # need at least 2 sentences so that one can be the target sentence
        while running_context_length > max_context_length and len(to_be_processed) > 1:
          if len(words) < 3:
            print_msg('-----------------------------------------------', verbose_level = 2)
            print_msg('INFO: Skipping {}'.format(to_be_processed[0]), verbose_level = 2)
            print_msg('INFO: Running Sequence Length = {}'.format(running_context_length), verbose_level = 2)
            print_msg('INFO: Target sentence {} should contain at least 3 tokens: <query> <answer> <period> but only has {}'.format(line.strip(), len(words)), verbose_level = 2)
            break
          else:
            data['context_length'].append(running_context_length)
            data['target_length'].append(len(words) - 1) # exclude last token which is most likely a period

            running_context_length -= len(to_be_processed[0])
            num_lines_processed += 1
            del to_be_processed[0]
        for word in words:
          if word not in self.dictionary.word2idx:
            word = '<unk>'
          data['data'].append(self.dictionary.word2idx[word])
          self.dictionary.update_count(word)

        running_context_length += len(words)

    print_msg('Processed {} out of total {}'.format(num_lines_processed, num_lines_in_file), verbose_level = 2)
    while num_lines_processed < num_lines_in_file:
      data['context_length'].append(0)
      data['target_length'].append(0)
      num_lines_processed += 1


  def tokenize_test_file(self, file, max_context_length, data):
    # Tokenize file content
    with open(file, 'r') as f:
      for line in f:
        words = line.split()
        last_period = rindex(words, '.')
        last_stopping_token = last_period if last_period > 0 else max(rindex(words, '!'), rindex(words, '?'))
        context_length = last_stopping_token + 1
        target_length = len(words) - context_length

        assert target_length > 0, "Target length must be positive"

        data['offsets'].append(len(data['data']) + 1)
        data['context_length'].append(context_length)
        data['target_length'].append(target_length)

        for word in words:
          if word not in self.dictionary.word2idx:
            word = '<unk>'
          data['data'].append(self.dictionary.word2idx[word])
          self.dictionary.update_count(word)


def debug_translate(word2idx):
  with h5py.File('lambada-ar.hdf5', "r") as f:
    train = {
      'data': f['train_data'],
      'offsets': f['train_offsets'],
      'context_length': f['train_context_length'],
      'target_length': f['train_target_length']
    }
    test = {
      'data': f['test_data'],
      'offsets': f['test_offsets'],
      'context_length': f['test_context_length'],
      'target_length': f['test_target_length']
    }
    valid = {
      'data': f['valid_data'],
      'offsets': f['valid_offsets'],
      'context_length': f['valid_context_length'],
      'target_length': f['valid_target_length']
    }
    control = {
      'data': f['control_data'],
      'offsets': f['control_offsets'],
      'context_length': f['control_context_length'],
      'target_length': f['control_target_length']
    }

    file_type = raw_input('Which dataset to translate? (train/test/valid/control, default = train): ')
    to_translate = train
    if file_type == 'test':
      to_translate = test
    elif file_type == 'valid':
      to_translate = valid
    elif file_type == 'control':
      to_translate = control

    view_order = raw_input('View data from beginning or end? (begin/end, default = begin): ')

    assert len(to_translate['offsets']) == len(to_translate['context_length']), "Inconsistent count"
    assert len(to_translate['offsets']) == len(to_translate['target_length']), "Inconsistent count"

    for i in range(len(to_translate['offsets'])):
      index = raw_input('Enter a 1-based line index to view (default = next index): ')
      if index == 'q' or index == 'quit':
        break

      view_index = int(index) - 1 if index != '' else i if view_order == 'begin' or view_order == '' else len(to_translate['offsets']) - i - 1
      offset = to_translate['offsets'][view_index] - 1 # offset is 1-based index
      context_length = to_translate['context_length'][view_index]
      target_length = to_translate['target_length'][view_index]
      context = to_translate['data'][offset : offset + context_length]
      target = to_translate['data'][offset + context_length: offset + context_length + target_length]

      print '1-BASED LINE INDEX = {}'.format(view_index + 1)
      print('CONTEXT')
      print([word2idx[token] for token in context])

      print('TARGET')
      print([word2idx[token] for token in target])
      


def main(arguments):
  global args
  parser = argparse.ArgumentParser(
      description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('--data', type=str, default='/mnt/c/Users/lhoang/Dropbox/Personal/Work/lambada-dataset/',
                      help='location of the data corpus')
  parser.add_argument('--train', type=str, default='train-novels',
                      help='relative location (file or folder) of training data')
  parser.add_argument('--valid', type=str, default='lambada_development_plain_text.txt',
                      help='relative location (file or folder) of validation data')
  parser.add_argument('--test', type=str, default='lambada_test_plain_text.txt',
                      help='relative location (file or folder) of testing data')
  parser.add_argument('--control', type=str, default='lambada_control_test_data_plain_text.txt',
                      help='relative location (file or folder) of control data')
  parser.add_argument('--vocab', type=str, default='lambada_vocabulary_sorted.txt',
                      help='relative location of vocab file')
  parser.add_argument('--max_context_length', type=int, default=50,
                      help='max # of tokens to include in context')
  parser.add_argument('--debug_translate', type=bool, default=False,
                      help='translate the preprocessed .hdf5 back into words')
  parser.add_argument('--verbose_level', type=int, default=1,
                      help='level of verbosity, ranging from 0 to 2, default = 1')
  args = parser.parse_args(arguments)

  # get embeddings
  # word_to_idx, suffix_to_idx, prefix_to_idx, embeddings = get_vocab_embedding(args.vocabsize)

  corpus = Corpus(args.data, args.vocab)
  if args.debug_translate > 0:
    debug_translate(corpus.dictionary.idx2word)
  else:
    corpus.load(args.data, args.train, args.valid, args.test, args.control, args.max_context_length)
    corpus.dictionary.write_to_file('lambada-ar.vocab')
    with h5py.File('lambada-ar.hdf5', "w") as f:
      f['train_data']           = np.array(corpus.train['data'])
      f['train_offsets']        = np.array(corpus.train['offsets'])
      f['train_context_length'] = np.array(corpus.train['context_length'])
      f['train_target_length']  = np.array(corpus.train['target_length'])

      f['valid_data']           = np.array(corpus.valid['data'])
      f['valid_offsets']        = np.array(corpus.valid['offsets'])
      f['valid_context_length'] = np.array(corpus.valid['context_length'])
      f['valid_target_length']  = np.array(corpus.valid['target_length'])

      f['test_data']           = np.array(corpus.test['data'])
      f['test_offsets']        = np.array(corpus.test['offsets'])
      f['test_context_length'] = np.array(corpus.test['context_length'])
      f['test_target_length']  = np.array(corpus.test['target_length'])

      f['control_data']           = np.array(corpus.control['data'])
      f['control_offsets']        = np.array(corpus.control['offsets'])
      f['control_context_length'] = np.array(corpus.control['context_length'])
      f['control_target_length']  = np.array(corpus.control['target_length'])


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
