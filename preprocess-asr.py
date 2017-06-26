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

end_words  = { "?", "??", "???", "!", "!!", "!!!", ".", "?!", "!?" }
GLOVE_DIM = 100

# Preprocessing: split each training example from the dataset https://arxiv.org/abs/1610.08431 into context and target pairs.

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
  def __init__(self):
    self.dictify()

  def __init__(self, vocab_file, punc_file, stop_word_file):
    self.dictify(vocab_file, None, None, punc_file, stop_word_file)

  def __init__(self, glove_file, glove_size, punc_file, stop_word_file):
    self.dictify(None, glove_file, glove_size, punc_file, stop_word_file)

  def dictify(self):
    self.dictionary = Dictionary()
    self.punctuations = []
    self.dictionary.add_word('<sep>') # map to 0 for masked rnn
    self.dictionary.add_word('<unk>')


  def dictify(self, vocab_file, glove_file, glove_size, punc_file, stop_word_file):
    self.dictionary = Dictionary()
    self.punctuations = []
    self.stopwords = []
    self.dictionary.add_word('<sep>') # map to 0 for masked rnn
    self.dictionary.add_word('<unk>')

    if vocab_file != None:
      with open(vocab_file, 'r') as f:
        for line in f:
          if line.strip():
            self.dictionary.add_word(line.strip())
    elif glove_file != None:
      self.embeddings = [np.zeros(GLOVE_DIM) for _ in range(len(self.dictionary))]
      with codecs.open(glove_file, "r", encoding="utf-8") as gf:
        num_glove = 0
        for line in gf:
          tokens = line.split(' ')
          self.dictionary.add_word(tokens[0])
          self.embeddings.append(np.array(tokens[1:]).astype(float))
          num_glove += 1
          if num_glove == glove_size:
            break
    else:
      raise Exception('Either vocab file or glove file must be specified.')

    with open(punc_file, 'r') as f:
      for line in f:
        if line.strip():
          self.punctuations.append(self.dictionary.add_word(line.strip()))

    with open(stop_word_file, 'r') as f:
      for line in f:
        if line.strip():
          self.stopwords.append(self.dictionary.add_word(line.strip()))

    print 'Vocab size = {}'.format(len(self.dictionary), verbose_level = 1)

  def load(self, path, train, valid, test, control):
    self.train   = self.tokenize(os.path.join(path, train),   training = True)
    self.valid   = self.tokenize(os.path.join(path, valid),   training = True)
    self.test    = self.tokenize(os.path.join(path, test),    training = False)
    self.control = self.tokenize(os.path.join(path, control), training = False)

    print_msg('\nTraining Data Statistics:\n', verbose_level = 1)
    train_context_length = self.train['location'][:,1]
    train_context_length = train_context_length[train_context_length > 0]
    print_msg('Context Length: max = {}, min = {}, average = {}, std = {}'.format(
      np.max(train_context_length), np.min(train_context_length), np.mean(train_context_length), np.std(train_context_length)), verbose_level = 1)

    train_target_length = self.train['location'][:,2]
    train_target_length = train_target_length[train_target_length > 0]
    print_msg('Target Length: max = {}, min = {}, average = {}, std = {}'.format(
      np.max(train_target_length), np.min(train_target_length), np.mean(train_target_length), np.std(train_target_length)), verbose_level = 1)


  def tokenize(self, path, training):
    assert os.path.exists(path)
    
    data = { 
      'data': [], # token ids for each word in the corpus 
      'offsets': [], # offset locations for each line in the final 1-d data array 
      'context_length': [], # count of words in the context (excluding target)
      'target_length': [] # count of words in the target
    }

    self.tokenize_file(path, data, training)

    sorted_data = { 'data': data['data'] }

    loc = np.array([np.array(data['offsets']), np.array(data['context_length']), np.array(data['target_length'])]).T
    loc = loc[np.argsort(-loc[:,1])] # sort by context length in descending order
    sorted_data['location'] = loc
    return sorted_data

 
  # update the ids, offsets, word counts, line counts
  def tokenize_file(self, file, data, training):
    num_lines_in_file = 0
    with open(file, 'r') as f:
      for line in f:
        num_lines_in_file += 1
        words = line.split()
        num_words = len(words)

        sep = -1
        for i in range(num_words - 2, -1, -1):
          if words[i] in end_words:
            sep = i
            break

        if training:
          # make sure answer is part of context (for computing loss & gradients during training)
          found_answer = False
          answer = words[num_words - 1]
          for i in range(0, sep + 1):
            if answer == words[i]:
              found_answer = True
          if not found_answer:
            print_msg('INFO: SKIPPING... Target answer not found in context', verbose_level = 2)
            continue

        target_length = num_words - sep - 1
        if target_length < 3:
          print_msg('INFO: SKIPPING... Target sentence should contain at least 3 tokens', verbose_level = 2)
          continue

        data['offsets'].append(len(data['data']) + 1)
        data['context_length'].append(sep + 1)
        data['target_length'].append(num_words - sep - 1)

        for word in words:
          if word not in self.dictionary.word2idx:
            word = '<unk>'
          data['data'].append(self.dictionary.word2idx[word])
          self.dictionary.update_count(word)

        print_msg('Processed {} lines'.format(num_lines_in_file), verbose_level = 3)


def debug_translate(idx2word, mode):
  if mode == 'manual':
    input = raw_input('Enter sentence to translate: ')
    while input != 'q' and input != 'quit':
      tokens = input.split()
      print(' '.join([idx2word[int(t)] for t in tokens]))
      input = raw_input('Enter sentence to translate: ')
  else:
    with h5py.File('lambada-asr.hdf5', "r") as f:
      train = {
        'data': f['train_data'],
        'location': np.array(f['train_location'], dtype=int),
      }
      test = {
        'data': f['test_data'],
        'location': np.array(f['test_location'], dtype=int),
      }
      valid = {
        'data': f['valid_data'],
        'location': np.array(f['valid_location'], dtype=int),
      }
      control = {
        'data': f['control_data'],
        'location': np.array(f['control_location'], dtype=int),
      }

      file_type = raw_input('Which dataset to translate? (train/test/valid/control, default = train): ')
      to_translate = train
      if file_type == 'test':
        to_translate = test
      elif file_type == 'valid':
        to_translate = valid
      elif file_type == 'control':
        to_translate = control

      # sort by increasing offset to view sequentially (easier for debug)
      to_translate['location'] = to_translate['location'][np.argsort(to_translate['location'][:,0])]
      num_examples = to_translate['location'].shape[0]

      view_order = raw_input('View data from beginning or end? (begin/end, default = begin): ')

      for i in range(num_examples):
        index = raw_input('Enter a 1-based line index to view (default = next index): ')
        if index == 'q' or index == 'quit':
          break

        view_index = int(index) - 1 if index != '' else i if view_order == 'begin' or view_order == '' else num_examples - i - 1
        offset = to_translate['location'][view_index,0] - 1 # offset is 1-based index
        context_length = to_translate['location'][view_index,1]
        target_length = to_translate['location'][view_index,2]
        context = to_translate['data'][offset : offset + context_length]
        target = to_translate['data'][offset + context_length: offset + context_length + target_length]

        print '1-BASED LINE INDEX = {}'.format(view_index + 1)
        print('CONTEXT')
        print([idx2word[token] for token in context])

        print('TARGET')
        print([idx2word[token] for token in target])
      

def main(arguments):
  global args
  parser = argparse.ArgumentParser(
      description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('--data', type=str, default='/mnt/c/Users/lhoang/Dropbox/Personal/Work/lambada-dataset/lambada-train-valid/original/',
                      help='location of the data corpus')
  parser.add_argument('--glove', type=str, default='', # e.g. data/glove.6B.100d.txt
                      help='absolute path of the glove embedding file')
  parser.add_argument('--glove_size', type=int, default=-1,
                      help='size of the vocab to build from glove embeddings')
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
  parser.add_argument('--punctuations', type=str, default='punctuations.txt',
                      help='relative location of punctuation file')
  parser.add_argument('--stopwords', type=str, default='mctest-stopwords.txt',
                      help='relative location of stop-words file')
  parser.add_argument('--out_file', type=str, default='lambada-asr.hdf5',
                      help='output hdf5 file')
  parser.add_argument('--debug_translate', type=str, default='',
                      help='translate the preprocessed .hdf5 back into words, or "manual" to translate manual input')
  parser.add_argument('--verbose_level', type=int, default=2,
                      help='level of verbosity, ranging from 0 to 2, default = 2')
  args = parser.parse_args(arguments)
  # get embeddings
  # word_to_idx, suffix_to_idx, prefix_to_idx, embeddings = get_vocab_embedding(args.vocabsize)

  if len(args.glove) > 0:
    corpus = Corpus(args.glove, args.glove_size, args.data + args.punctuations, args.data + args.stopwords)
  else:
    corpus = Corpus(args.data + args.vocab, args.data + args.punctuations, args.data + args.stopwords)
  if len(args.debug_translate) > 0:
    debug_translate(corpus.dictionary.idx2word, args.debug_translate)
  else:
    corpus.load(args.data, args.train, args.valid, args.test, args.control)
    with h5py.File(args.out_file, "w") as f:
      f['punctuations']     = np.array(corpus.punctuations) # punctuations are ignored during test time
      f['stopwords']        = np.array(corpus.stopwords) # punctuations are ignored during test time
      f['vocab_size']       = np.array([len(corpus.dictionary)])

      f['train_data']       = np.array(corpus.train['data'])
      f['train_location']   = np.array(corpus.train['location'])

      f['valid_data']       = np.array(corpus.valid['data'])
      f['valid_location']   = np.array(corpus.valid['location'])

      f['test_data']        = np.array(corpus.test['data'])
      f['test_location']    = np.array(corpus.test['location'])

      f['control_data']     = np.array(corpus.control['data'])
      f['control_location'] = np.array(corpus.control['location'])

      if corpus.embeddings != None:
        f['word_embeddings'] = np.array(corpus.embeddings)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
