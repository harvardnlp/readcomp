#!/usr/bin/env python

import numpy as np
from numpy import linalg as LA
import nltk
from nltk.corpus import wordnet as wn
import sys
import argparse
import codecs

args = {}

def main(arguments):
  global args
  parser = argparse.ArgumentParser(
      description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter) 
  parser.add_argument('--glove', type=str, default='',
                      help='absolute path of the glove embedding file')

  args = parser.parse_args(arguments)

  embeddings = {}
  with codecs.open(args.glove, "r", encoding="utf-8") as gf:
    for line in gf:
      tokens = line.split(' ')
      embeddings[tokens[0].strip()] = np.array(tokens[1:]).astype(float)

  with codecs.open(args.glove + ".avg", "w", encoding="utf-8") as wf:
    for word in embeddings:
      out_emb = embeddings[word]
      word_syn = wn.synsets(word)
      if len(word_syn) > 0:
        word_def = word_syn[0].definition()
        word_def_tok = nltk.word_tokenize(word_def)
        num_tok = 1
        for wt in word_def_tok:
          if wt in embeddings:
            out_emb += embeddings[wt]
            num_tok += 1
        out_emb = out_emb / float(num_tok)

      wf.write(u'{} {}\n'.format(word, ' '.join([str(oe) for oe in out_emb])))



  # input = raw_input('Enter two tokens separated by space: ')
  # while input != 'q' and input != 'quit':
  #   tokens = input.split()
  #   print(np.dot(embeddings[tokens[0]], embeddings[tokens[1]]) / (
  #     LA.norm(embeddings[tokens[0]]) * np.linalg.norm(embeddings[tokens[1]])))
  #   input = raw_input('Enter two tokens separated by space: ')


if __name__ == '__main__':
  sys.exit(main(sys.argv[1:]))
