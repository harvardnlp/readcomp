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

# Preprocessing: split each training example from the dataset https://arxiv.org/abs/1610.08431 into context and target pairs.

def rindex(mylist, myvalue):
  try:
    index = len(mylist) - mylist[::-1].index(myvalue) - 1
  except:
    index = -1
  return index


def inter_translate(idx2word):
  input = raw_input('Enter sentence to translate: ')
  while input != 'q' and input != 'quit':
    tokens = input.split()
    print(' '.join([idx2word[int(t)] for t in tokens]))
    input = raw_input('Enter sentence to translate: ')


def load_file(file):
  with h5py.File(file, "r") as f:
    train = {
      'data': np.array(f['train_data']),
      'pref': np.array(f['train_pref']),
      'suff': np.array(f['train_suff']),
      'post': np.array(f['train_post']),
      'location': np.array(f['train_location'], dtype=int),
    }
    test = {
      'data': np.array(f['test_data'], dtype=int),
      'pref': np.array(f['test_pref'], dtype=int),
      'suff': np.array(f['test_suff'], dtype=int),
      'post': np.array(f['test_post'], dtype=int),
      'location': np.array(f['test_location'], dtype=int),
    }
    valid = {
      'data': np.array(f['valid_data'], dtype=int),
      'pref': np.array(f['valid_pref'], dtype=int),
      'suff': np.array(f['valid_suff'], dtype=int),
      'post': np.array(f['valid_post'], dtype=int),
      'location': np.array(f['valid_location'], dtype=int),
    }
    control = {
      'data': np.array(f['control_data'], dtype=int),
      'pref': np.array(f['control_pref'], dtype=int),
      'suff': np.array(f['control_suff'], dtype=int),
      'post': np.array(f['control_post'], dtype=int),
      'location': np.array(f['control_location'], dtype=int),
    }
  return train,test,valid,control


def validate_tensor(corpus, t):
  num_examples = t['location'].shape[0]
  idx = np.random.randint(num_examples, size=(1000))
  for i in idx:
    offset = t['location'][i,0] - 1 # offset is 1-based index
    context_length = t['location'][i,1]
    target_length = t['location'][i,2]
    context = t['data'][offset : offset + context_length]
    target = t['data'][offset + context_length: offset + context_length + target_length]
    text = t['data'][offset : offset + context_length + target_length]

    doc_words = [corpus.dictionary.idx2word[token_id] for token_id in text]

    idx2pref = {v: k for k, v in corpus.dictionary.pref2idx.iteritems()}
    idx2suff = {v: k for k, v in corpus.dictionary.suff2idx.iteritems()}
    idx2post = {v: k for k, v in corpus.dictionary.post2idx.iteritems()}

    expected_pref = [get_prefix(w) for w in doc_words]
    expected_suff = [get_suffix(w) for w in doc_words]
    expected_post = [tag[1] for tag in nltk.pos_tag(doc_words)]

    actual_pref = [idx2pref[token_id] for token_id in t['pref'][offset : offset + context_length + target_length]]
    actual_suff = [idx2suff[token_id] for token_id in t['suff'][offset : offset + context_length + target_length]]
    actual_post = [idx2post[token_id] for token_id in t['post'][offset : offset + context_length + target_length]]

    assert np.array_equal(expected_pref, actual_pref), 'prefix validation failed, i = {}\nwords={}\ne = {}\na = {}'.format(
      str(i), ' '.join(doc_words), ' '.join(str(p) for p in expected_pref), ' '.join(str(p) for p in actual_pref))

    assert np.array_equal(expected_suff, actual_suff), 'suffix validation failed, i = {}\nwords={}\ne = {}\na = {}'.format(
      str(i), ' '.join(doc_words), ' '.join(str(p) for p in expected_suff), ' '.join(str(p) for p in actual_suff))

    # pos tags are performed before <unk> tokenization, allow for 5% mismatch
    num_mis = 0
    for j in range(len(expected_post)):
      if doc_words[j] != UNKNOWN and expected_post[j] != actual_post[j]:
        expected_post[j] = '*' + expected_post[j] + '*'
        actual_post[j] = '*' + actual_post[j] + '*'
        num_mis += 1

    if num_mis * 100.0 / len(expected_post) > 5:
      print('pos tags validation failed, i = {}\nwords={}\ne = {}\na = {}'.format(
        str(i), ' '.join(doc_words), ' '.join(str(p) for p in expected_post), ' '.join(str(p) for p in actual_post)))
      sys.exit(0)


def validate(corpus, file):
  train,test,valid,control = load_file(file)
  print('NOTE: this does not validate extra features such as token frequency etc...')
  validate_tensor(corpus, train)
  validate_tensor(corpus, test)
  validate_tensor(corpus, valid)
  validate_tensor(corpus, control)
  print('Validation Passed')


def debug_translate(corpus, file, mode):
  if mode == 'text':
    inter_translate(corpus.dictionary.idx2word)
  elif mode == 'suffix':
    inter_translate({v: k for k, v in corpus.dictionary.suff2idx.iteritems()})
  elif mode == 'prefix':
    inter_translate({v: k for k, v in corpus.dictionary.pref2idx.iteritems()})
  elif mode == 'postag':
    inter_translate({v: k for k, v in corpus.dictionary.post2idx.iteritems()})
  else:
    train,test,valid,control = load_file(file)

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
  parser.add_argument('--data', type=str, default='~/Dropbox/Personal/Work/lambada-dataset/lambada-train-valid/original/',
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
  parser.add_argument('--extra_vocab', type=str, default=None,
                      help='relative location of extra vocab file to load into dictionary')
  parser.add_argument('--punctuations', type=str, default='punctuations.txt',
                      help='relative location of punctuation file')
  parser.add_argument('--stopwords', type=str, default='mctest-stopwords.txt',
                      help='relative location of stop-words file')
  parser.add_argument('--context_query_separator', type=str, default='',
                      help='separator token between context, query and answer, for CNN dataset use $$$')
  parser.add_argument('--answer_identifier', type=str, default='',
                      help='identifier for answer token in the query, for CNN dataset this is @placeholder')
  parser.add_argument('--out_file', type=str, default='lambada-asr.hdf5',
                      help='output hdf5 file')
  parser.add_argument('--debug_translate', type=str, default='',
                      help='translate the preprocessed .hdf5 back into words, or "manual" to translate manual input')
  parser.add_argument('--verbose_level', type=int, default=2,
                      help='level of verbosity, ranging from 0 to 2, default = 2')
  parser.add_argument('--validate', action='store_true',
                      help='whether to validate values in the output .hdf5 file')
  args = parser.parse_args(arguments)
  # get embeddings
  # word_to_idx, suffix_to_idx, prefix_to_idx, embeddings = get_vocab_embedding(args.vocabsize)


  out_vocab_file_prefix = args.out_file.split('.')[0]
  start_time = time.time()
  if len(args.debug_translate) or args.validate:
    corpus = datamodel.Corpus(args.verbose_level, None, None, None, None, None, None, None, None)
    corpus.load_vocab(out_vocab_file_prefix)
    if args.validate:
      validate(corpus, args.out_file)
    else:
      debug_translate(corpus, args.out_file, args.debug_translate)
  else:
    if len(args.glove):
      corpus = datamodel.Corpus(args.verbose_level, None, args.glove, args.glove_size, args.data + args.punctuations, 
        args.data + args.stopwords, args.data + args.extra_vocab if args.extra_vocab else None, args.context_query_separator, args.answer_identifier)
    else:
      corpus = datamodel.Corpus(args.verbose_level, args.data + args.vocab, None, None, args.data + args.punctuations, 
        args.data + args.stopwords, args.data + args.extra_vocab if args.extra_vocab else None, args.context_query_separator, args.answer_identifier)
    corpus.load(args.data, args.train, args.valid, args.test, args.control)
    corpus.save(out_vocab_file_prefix)

    with h5py.File(args.out_file, "w") as f:
      f['punctuations']     = np.array(corpus.punctuations) # punctuations are ignored during test time
      f['stopwords']        = np.array(corpus.stopwords) # punctuations are ignored during test time
      f['vocab_size']       = np.array([len(corpus.dictionary)])
      f['pref_vocab_size']  = np.array([len(corpus.dictionary.pref2idx)])
      f['suff_vocab_size']  = np.array([len(corpus.dictionary.suff2idx)])
      f['post_vocab_size']  = np.array([len(corpus.dictionary.post2idx)])

      f['train_data']       = np.array(corpus.train['data'])
      f['train_pref']       = np.array(corpus.train['pref'])
      f['train_suff']       = np.array(corpus.train['suff'])
      f['train_post']       = np.array(corpus.train['post'])
      f['train_extr']       = np.array(corpus.train['extr'])
      f['train_location']   = np.array(corpus.train['location'])

      f['valid_data']       = np.array(corpus.valid['data'])
      f['valid_pref']       = np.array(corpus.valid['pref'])
      f['valid_suff']       = np.array(corpus.valid['suff'])
      f['valid_post']       = np.array(corpus.valid['post'])
      f['valid_extr']       = np.array(corpus.valid['extr'])
      f['valid_location']   = np.array(corpus.valid['location'])

      f['test_data']        = np.array(corpus.test['data'])
      f['test_pref']        = np.array(corpus.test['pref'])
      f['test_suff']        = np.array(corpus.test['suff'])
      f['test_post']        = np.array(corpus.test['post'])
      f['test_extr']        = np.array(corpus.test['extr'])
      f['test_location']    = np.array(corpus.test['location'])

      f['control_data']     = np.array(corpus.control['data'])
      f['control_pref']     = np.array(corpus.control['pref'])
      f['control_suff']     = np.array(corpus.control['suff'])
      f['control_post']     = np.array(corpus.control['post'])
      f['control_extr']     = np.array(corpus.control['extr'])
      f['control_location'] = np.array(corpus.control['location'])

      if corpus.embeddings != None:
        f['word_embeddings'] = np.array(corpus.embeddings)
  end_time = time.time()
  datamodel.print_msg('Total elapsed time = {}s'.format(str(end_time - start_time)), 1, args.verbose_level)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
