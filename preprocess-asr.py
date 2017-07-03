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

args = {}

end_words  = { "?", "??", "???", "!", "!!", "!!!", ".", "?!", "!?" }

GLOVE_DIM = 100
SEPARATOR = '<sep>'
UNKNOWN = '<unk>'

# Preprocessing: split each training example from the dataset https://arxiv.org/abs/1610.08431 into context and target pairs.

def get_suffix(w):
    if len(w) < 2:
        return w
    return w[-2:]


def get_prefix(w):
    if len(w) < 2:
        return w
    return w[:2]


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
    self.word2count = {} # for NCE if training LM
    self.idx2word = []

    self.pref2idx = {}
    self.suff2idx = {}
    self.post2idx = {} # pos tags
    self.punc2idx = {} # punctuations
    self.stop2idx = {} # stop words


  def add_word(self, word):
    if word not in self.word2idx:
      self.idx2word.append(word)
      self.word2idx[word] = len(self.idx2word) - 1 # subtract 1 to make <sep> token index 0
      self.word2count[word] = 0 # set to 0 since a word in vocab may not appear in training data

    suff = get_suffix(word)
    if suff not in self.suff2idx:
      self.suff2idx[suff] = len(self.suff2idx) + 1

    pref = get_prefix(word)
    if pref not in self.pref2idx:
      self.pref2idx[pref] = len(self.pref2idx) + 1

    return self.word2idx[word]


  def add_pos_tag(self, tag):
    if tag not in self.post2idx:
      self.post2idx[tag] = len(self.post2idx) + 1
    return self.post2idx[tag]


  def update_count(self, word):
    self.word2count[word] = self.word2count[word] + 1 # the word must be part of the vocab

  def __len__(self):
    return len(self.idx2word)

  def write_to_file(self, file_prefix):
    with codecs.open(file_prefix + '.vocab', 'w', encoding='utf8') as outf:
      for i in range(len(self.idx2word)):
        word = self.idx2word[i]
        count = self.word2count[word]
        outf.write(u'{}\t{}\t{}\n'.format(i,word,count))

    with codecs.open(file_prefix + '.prefix.vocab', 'w', encoding='utf8') as pref:
      for key, value in sorted(self.pref2idx.iteritems(), key=lambda (k,v): (v,k)):
        pref.write(u'{}\t{}\n'.format(key,value))

    with codecs.open(file_prefix + '.suffix.vocab', 'w', encoding='utf8') as suff:
      for key, value in sorted(self.suff2idx.iteritems(), key=lambda (k,v): (v,k)):
        suff.write(u'{}\t{}\n'.format(key,value))

    with codecs.open(file_prefix + '.pos.vocab', 'w', encoding='utf8') as posf:
      for key, value in sorted(self.post2idx.iteritems(), key=lambda (k,v): (v,k)):
        posf.write(u'{}\t{}\n'.format(key,value))


  def read_from_file(self, file_prefix):
    with codecs.open(file_prefix + '.vocab', 'r', encoding='utf8') as inf:
      for line in inf:
        parts = line.split()
        self.word2idx[parts[1]] = int(parts[0])
        self.idx2word.append(parts[1])
        self.word2count[parts[1]] = int(parts[2])

    with codecs.open(file_prefix + '.prefix.vocab', 'r', encoding='utf8') as pref:
      for line in pref:
        parts = line.split()
        self.pref2idx[parts[0]] = int(parts[1])

    with codecs.open(file_prefix + '.suffix.vocab', 'r', encoding='utf8') as suff:
      for line in suff:
        parts = line.split()
        self.suff2idx[parts[0]] = int(parts[1])

    with codecs.open(file_prefix + '.pos.vocab', 'r', encoding='utf8') as posf:
      for line in posf:
        parts = line.split()
        self.post2idx[parts[0]] = int(parts[1])


class Corpus(object):
  def __init__(self, vocab_file, glove_file, glove_size, punc_file, stop_word_file):
    self.puncstop_answer_count = 0
    self.dictify(vocab_file, glove_file, glove_size, punc_file, stop_word_file)


  def dictify(self, vocab_file, glove_file, glove_size, punc_file, stop_word_file):
    self.dictionary = Dictionary()

    if vocab_file != None or glove_file != None:
      self.dictionary.add_word(SEPARATOR) # map to 0 for masked rnn
      self.dictionary.add_word(UNKNOWN)
      if vocab_file != None:
        with open(vocab_file, 'r') as f:
          for line in f:
            if line.strip():
              self.dictionary.add_word(line.strip())
      else:
        print_msg('Loading GLOVE ...', verbose_level = 1)
        self.embeddings = [np.random.rand(GLOVE_DIM) * 0.1 for _ in range(len(self.dictionary))]
        with codecs.open(glove_file, "r", encoding="utf-8") as gf:
          num_glove = 0
          for line in gf:
            tokens = line.split(' ')
            self.dictionary.add_word(tokens[0])
            self.embeddings.append(np.array(tokens[1:]).astype(float))
            num_glove += 1
            if num_glove == glove_size:
              break
        print_msg('Done ...', verbose_level = 1)

    if len(self.dictionary) > 0:
      self.punctuations = []
      self.stopwords = []

      with open(punc_file, 'r') as f:
        for line in f:
          punc = line.strip()
          if punc:
            self.punctuations.append(self.dictionary.add_word(punc))
            if punc not in self.dictionary.punc2idx:
              self.dictionary.punc2idx[punc] = len(self.dictionary.punc2idx) + 1

      with open(stop_word_file, 'r') as f:
        for line in f:
          sw = line.strip()
          if sw:
            self.stopwords.append(self.dictionary.add_word(sw))
            if sw not in self.dictionary.stop2idx:
              self.dictionary.stop2idx[sw] = len(self.dictionary.stop2idx) + 1

      print 'Vocab size = {}'.format(len(self.dictionary), verbose_level = 1)


  def load_vocab(self, vocab_file_prefix):
    print_msg('Loading vocab...', verbose_level = 1)
    self.dictionary.read_from_file(vocab_file_prefix)


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

    print_msg('\nPrefix and Suffix Statistics:', verbose_level = 1)
    print_msg('Prefix Size: {}'.format(len(self.dictionary.pref2idx)), verbose_level = 1)
    print_msg('Suffix Size: {}'.format(len(self.dictionary.suff2idx)), verbose_level = 1)
    print_msg('POS Size: {}'.format(len(self.dictionary.post2idx)), verbose_level = 1)

    print_msg('\nCount of cases where answer is a punctuation symbol or stop word: ' + str(self.puncstop_answer_count), verbose_level = 1)


  def save(self, file_prefix):
    self.dictionary.write_to_file(file_prefix)


  def tokenize(self, path, training):
    assert os.path.exists(path)
    
    data = { 
      'data': [], # token ids for each word in the corpus 
      'pref': [], # prefix ids 
      'suff': [], # suffix ids 
      'post': [], # pos tags 
      'extr': [], # extra features, such as frequency of token in the context, whether previous bi-gram of token match with that of the answer etc...
      'offsets': [], # offset locations for each line in the final 1-d data array 
      'context_length': [], # count of words in the context (excluding target)
      'target_length': [] # count of words in the target
    }

    self.tokenize_file(path, data, training)

    sorted_data = { 'data': data['data'], 'pref': data['pref'], 'suff': data['suff'], 'post': data['post'], 'extr': data['extr'] }

    loc = np.array([np.array(data['offsets']), np.array(data['context_length']), np.array(data['target_length'])]).T
    loc = loc[np.argsort(-loc[:,1])] # sort by context length in descending order
    sorted_data['location'] = loc
    return sorted_data

 
  # update the ids, offsets, word counts, line counts
  def tokenize_file(self, file, data, training):
    num_lines_in_file = 0
    with codecs.open(file, 'r', encoding='utf8') as f:
      for line in f:
        num_lines_in_file += 1
        words = line.split()
        num_words = len(words)

        pos_tags = [t[1] for t in nltk.pos_tag(words)]

        sep = -1 # last index of word in the context
        for i in range(num_words - 2, -1, -1):
          if words[i] in end_words:
            sep = i
            break

        if training:
          # make sure answer is part of context (for computing loss & gradients during training)
          found_answer = False
          answer = words[num_words - 1]
          if answer in self.dictionary.punc2idx or answer in self.dictionary.stop2idx:
            self.puncstop_answer_count += 1
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

        words = [word if word in self.dictionary.word2idx else UNKNOWN for word in words]

        extr_word_freq = {}
        for i in range(len(words)):
          word = words[i]

          if word not in extr_word_freq:
            extr_word_freq[word] = 0

          # only count within context for non-punctuation and non-stopword tokens
          if i <= sep and word not in self.dictionary.punc2idx and word not in self.dictionary.stop2idx:
            extr_word_freq[word] += 1

          data['data'].append(self.dictionary.word2idx[word])

          pref = get_prefix(word)
          data['pref'].append(self.dictionary.pref2idx[pref])

          suff = get_suffix(word)
          data['suff'].append(self.dictionary.suff2idx[suff])

          pos_tag = pos_tags[i]
          data['post'].append(self.dictionary.add_pos_tag(pos_tag))

          self.dictionary.update_count(word)

        previous_answer_bigram = [words[num_words - 3], words[num_words - 2]]
        for i in range(len(words)):
          word = words[i]
          extra_features = []

          freq = float(extr_word_freq[word]) / len(words)
          bigram_match = 1 if i > 2 and i <= sep and words[i - 2] == words[num_words - 3] and words[i - 1] == words[num_words - 2] else 0 
          
          extra_features.append(freq)
          extra_features.append(bigram_match)

          data['extr'].append(np.array(extra_features))

        print_msg('Processed {} lines'.format(num_lines_in_file), verbose_level = 3)


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
  parser.add_argument('--out_vocab_file_prefix', type=str, default='lambada-asr',
                      help='file name prefix for output vocab files')
  parser.add_argument('--debug_translate', type=str, default='',
                      help='translate the preprocessed .hdf5 back into words, or "manual" to translate manual input')
  parser.add_argument('--verbose_level', type=int, default=2,
                      help='level of verbosity, ranging from 0 to 2, default = 2')
  parser.add_argument('--validate', action='store_true',
                      help='whether to validate values in the output .hdf5 file')
  args = parser.parse_args(arguments)
  # get embeddings
  # word_to_idx, suffix_to_idx, prefix_to_idx, embeddings = get_vocab_embedding(args.vocabsize)

  start_time = time.time()
  if len(args.debug_translate) or args.validate:
    corpus = Corpus(None, None, None, None, None)
    corpus.load_vocab(args.out_vocab_file_prefix)
    if args.validate:
      validate(corpus, args.out_file)
    else:
      debug_translate(corpus, args.out_file, args.debug_translate)
  else:
    if len(args.glove):
      corpus = Corpus(None, args.glove, args.glove_size, args.data + args.punctuations, args.data + args.stopwords)
    else:
      corpus = Corpus(args.data + args.vocab, None, None, args.data + args.punctuations, args.data + args.stopwords)
    corpus.load(args.data, args.train, args.valid, args.test, args.control)
    corpus.save(args.out_vocab_file_prefix)

    with h5py.File(args.out_file, "w") as f:
      f['punctuations']     = np.array(corpus.punctuations) # punctuations are ignored during test time
      f['stopwords']        = np.array(corpus.stopwords) # punctuations are ignored during test time
      f['vocab_size']       = np.array([len(corpus.dictionary)])

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
  print_msg('Total elapsed time = {}s'.format(str(end_time - start_time)), verbose_level = 1)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
