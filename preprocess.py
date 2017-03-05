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
        print 'Vocab size = {}'.format(len(self.dictionary))


    def load(self, path, train, valid, test, control, max_seq_length = 100):
        self.train   = self.tokenize(os.path.join(path, train), max_seq_length)
        self.valid   = self.tokenize(os.path.join(path, valid), 0)
        self.test    = self.tokenize(os.path.join(path, test), 0)
        self.control = self.tokenize(os.path.join(path, control), 0)


    def tokenize(self, path, max_seq_length):
        """
        Tokenizes all data found recursively in a path (supports single file or folder).
        :param max_seq_length: Sentences will be concatenated up to this length.
            If a single sentence is longer it is unchanged.
            Set to 0 to treat each sentence separately. 
        """
        assert os.path.exists(path)
        
        if os.path.isdir(path):
            ids = []
            for root, dir_names, file_names in os.walk(path):
                for file in fnmatch.filter(file_names, '*.txt'):
                    self.tokenize_file(os.path.join(root, file), max_seq_length, ids)
        else:
            ids = []
            self.tokenize_file(path, max_seq_length, ids)
        return ids


    def tokenize_file(self, file, max_seq_length, out_ids):
        # Tokenize file content
        with open(file, 'r') as f:
            running_length = 0
            for line in f:
                words = line.split()
                running_length += len(words)
                if running_length > max_seq_length:
                    running_length = len(words)
                    if len(out_ids) > 0:
                        words = ['<sep>'] + words
                for word in words:
                    if word not in self.dictionary.word2idx:
                        word = '<unk>'
                    out_ids.append(self.dictionary.word2idx[word])
                    self.dictionary.update_count(word)


def debug_translate(file, word2idx):
    with h5py.File('lambada.hdf5', "r") as f:
        train   = f['train']
        test    = f['test']
        valid   = f['valid']
        control = f['control']

        file_type = raw_input('Which file to translate? (train/test/valid/control, default = train): ')
        to_translate = train
        if file_type == 'test':
            to_translate = test
        elif file_type == 'valid':
            to_translate = valid
        elif file_type == 'control':
            to_translate = control

        test_lines = []
        line = []
        for token in to_translate:
            if token == 0:
                print(line)
                raw_input('Press Enter to continue...')
                test_lines.append(line)
                line = []
            else:
                line.append(word2idx[token])
        if len(line) > 0:
            test_lines.append(line)


def main(arguments):
    global args
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', type=str, default='/mnt/c/Users/lhoang/Dropbox/Personal/Work/lambada-dataset/small/',
                        help='location of the data corpus')
    parser.add_argument('--train', type=str, default='train',
                        help='relative location (file or folder) of training data')
    parser.add_argument('--valid', type=str, default='lambada_development_plain_text.txt',
                        help='relative location (file or folder) of validation data')
    parser.add_argument('--test', type=str, default='lambada_test_plain_text.txt',
                        help='relative location (file or folder) of testing data')
    parser.add_argument('--control', type=str, default='lambada_control_test_data_plain_text.txt',
                        help='relative location (file or folder) of control data')
    parser.add_argument('--vocab', type=str, default='lambada_vocabulary_sorted.txt',
                        help='relative location of vocab file')
    parser.add_argument('--debug_translate', type=str, default='',
                        help='translate the preprocessed .hdf5 back into words')
    args = parser.parse_args(arguments)

    # get embeddings
    # word_to_idx, suffix_to_idx, prefix_to_idx, embeddings = get_vocab_embedding(args.vocabsize)

    corpus = Corpus(args.data, args.vocab)
    if len(args.debug_translate) > 0:
        debug_translate(args.debug_translate, corpus.dictionary.idx2word)
    else:
        corpus.load(args.data, args.train, args.valid, args.test, args.control)
        corpus.dictionary.write_to_file('lambada.vocab')
        with h5py.File('lambada.hdf5', "w") as f:
            f['train']   = np.array(corpus.train)
            f['valid']   = np.array(corpus.valid)
            f['test']    = np.array(corpus.test)
            f['control'] = np.array(corpus.control)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
