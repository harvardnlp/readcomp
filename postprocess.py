#!/usr/bin/env python
from __future__ import unicode_literals

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

sys.stdout = codecs.getwriter('utf8')(sys.stdout)
sys.stderr = codecs.getwriter('utf8')(sys.stderr)

args = {}


def compute_accuracy(args):
  total = 0
  total_context = 0 # answer is in context
  correct = 0
  processed_batches = {}
  for filename in glob.glob(args.model_dump_pattern):
    if args.ensemble:
      inputs, outputs, predictions, answers, filename, batch_id = load_ensemble(filename, args.ensemble_type)
      if batch_id in processed_batches:
        continue
      else:
        processed_batches[batch_id] = True
    else:
      with h5py.File(filename, "r") as f:
        inputs      = np.array(f['inputs'],      dtype=int).T
        outputs     = np.array(f['outputs'],     dtype=float)
        predictions = np.array(f['predictions'], dtype=int)
        answers     = np.array(f['answers'],     dtype=int)

    batch_size, max_len = outputs.shape
    answer_index = answers > 0
    total += np.sum(answer_index)
    total_context += np.sum([1 if answers[b] > 0 and answers[b] in inputs[b] else 0 for b in range(batch_size)])
    correct += np.sum(predictions[answer_index] == answers[answer_index])

  print 'Correct = {}, Total = {}, % = {:.4f}%, Total Context = {}, % = {:.4f}%'.format(
    correct, total, float(correct) * 100 / total, total_context, float(correct) * 100 / total_context)


def max_attention_sum(ip, op):
  max_words = np.zeros(op.shape[0], dtype=int)
  word_to_prob = {}
  max_prob = 0

  for b in range(op.shape[0]):
    for w in range(op.shape[1]):
      word = ip[b][w]
      if word != 0:
        if word not in word_to_prob:
          word_to_prob[word] = 0
        word_to_prob[word] += op[b][w]
        if word_to_prob[word] > max_prob:
          max_prob = word_to_prob[word]
          max_words[b] = word

  return max_words


def load_ensemble(filename, vote_type):
  file_info = re.match('(.*)\.t7\.test\.(.*)\.dump', os.path.basename(filename))
  model_id = file_info.group(1)
  batch_id = file_info.group(2)
  ensemble_pattern = os.path.join(os.path.dirname(filename), '*.t7.test.{}.dump'.format(batch_id))
  filename = os.path.join(os.path.dirname(filename), 'ensemble.t7.test.{}.dump'.format(batch_id))

  inputs      = None
  answers     = None
  output_table = {}
  prediction_table = {}

  num_files = 0
  for filebatch in glob.glob(ensemble_pattern):
    num_files += 1
    with h5py.File(filebatch, "r") as f:
      file_inputs                 = np.array(f['inputs'],      dtype=int).T
      file_answers                = np.array(f['answers'],     dtype=int)
      output_table[num_files]     = np.array(f['outputs'],     dtype=float)
      prediction_table[num_files] = np.array(f['predictions'], dtype=int)

      if inputs is None or answers is None:
        inputs = file_inputs
        answers = file_answers
      else:
        assert (inputs == file_inputs).all(), "input is different for different model"
        assert (answers == file_answers).all(), "answers is different for different model"
  
  outputs = sum([output_table[key] for key in output_table]) / float(num_files)

  if vote_type == 'avg':
    predictions = max_attention_sum(inputs, outputs)
    print('predictions')
    print(predictions)

  elif vote_type == 'count':
    predictions = prediction_table[1]
    for b in range(prediction_table[1].shape[0]):
      word_to_count = {}
      max_count = 0
      for key in prediction_table:
        pred = prediction_table[key][b]
        if pred != 0:
          if pred not in word_to_count:
            word_to_count[pred] = 0
          word_to_count[pred] += 1
          if max_count < word_to_count[pred]:
            max_count = word_to_count[pred]
            predictions[b] = pred

  else:
    print('not implemented')
    os.exit(0)

  return inputs, outputs, predictions, answers, filename, int(batch_id)


def main(arguments):
  global args
  parser = argparse.ArgumentParser(
      description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter) 
  parser.add_argument('--model_dump_pattern', type=str, default='data\\dump\\*.dump',
                      help='file pattern of model dumps for analysis')
  parser.add_argument('--analysis_category_file', type=str, default='',
                      help='absolute path of analysis-category file')
  parser.add_argument('--vocab_file_prefix', type=str, default='lambada',
                      help='file name prefix of vocab files')
  parser.add_argument('--out_debug_file', type=str, default='debug.tsv',
                      help='create tsv debug files with answer rank, answer, prediction, and link to image')
  parser.add_argument('--in_context_only', action='store_true',
                      help='whether to process only instances where answer is in context')
  parser.add_argument('--incorrect_only', action='store_true',
                      help='whether to process only instances where answer is incorrect')
  parser.add_argument('--answer_rank', type=int, default=0,
                      help='only process instances where the answer is ranked at the specified number, 0 to print all')
  parser.add_argument('--ensemble', action='store_true',
                      help='whether to treat dump directory as ensemble dump, i.e. containing dump files for multiple models')
  parser.add_argument('--ensemble_type', type=str, default='avg',
                      help='type of ensemble voting, avg or count')
  parser.add_argument('--verbose_level', type=int, default=2,
                      help='level of verbosity, ranging from 0 to 2, default = 2')

  args = parser.parse_args(arguments)

  hfont = { 'fontname':'serif' }
  sns.set(font_scale=2,font='serif')
  plt.figure(figsize=(60, 40), dpi=100)

  compute_accuracy(args)

  corpus = datamodel.Corpus(args.verbose_level, None, None, None, None, None, None, None, None)
  corpus.load_vocab(args.vocab_file_prefix)
  idx2word = corpus.dictionary.idx2word
  word2idx = corpus.dictionary.word2idx

  category_labels = {}
  contexts = []

  analysis_category_correct = {}
  analysis_category_count = {}

  if len(args.analysis_category_file):

    with codecs.open(args.analysis_category_file, 'r', encoding='utf8') as acf:
      headers = acf.readline().split('\t')
      categories = []
      counts = []

      for h in range(1, len(headers)):
        hre = re.match('"(.*) ([0-9]+)"', headers[h])
        category_name = hre.group(1).strip()
        category_labels[category_name] = []
        categories.append(category_name)
        counts.append(int(hre.group(2).strip()))

        analysis_category_correct[category_name] = 0
        analysis_category_count[category_name] = 0

      for line in acf:
        values = line.split('\t')
        # hack: look up the example in analysis file using the first few words
        contexts.append(' '.join([ str(word2idx[datamodel.UNKNOWN]) if v not in word2idx else str(word2idx[v]) for v in values[0].strip().split()[:10] ]))

        assert len(values) - 1 == len(categories), 'number of values {} does not match number of headers {}'.format(len(values) - 1, len(categories))

        for i in range(1, len(values)):
          category_labels[categories[i - 1]].append(int(values[i].strip()))

      for i in range(len(categories)):
        assert np.sum(category_labels[categories[i]]) == counts[i], 'category count mismatch for category {}'.format(categories[i])

  if len(args.out_debug_file): 
    debug_file = codecs.open(args.out_debug_file, 'w', encoding='utf8')
    debug_file.write('ImageFile\tAnswer Rank\tAnswer\tPrediction\n')

  processed_batches = {}

  for filename in sorted(glob.glob(args.model_dump_pattern), reverse = True): # work on shorter examples first
    print 'Processing {}'.format(filename)

    if args.ensemble:
      inputs, outputs, predictions, answers, filename, batch_id = load_ensemble(filename, args.ensemble_type)
      if batch_id in processed_batches:
        continue
      else:
        processed_batches[batch_id] = True

    else:
      with h5py.File(filename, "r") as f:
        inputs      = np.array(f['inputs'],      dtype=int).T
        outputs     = np.array(f['outputs'],     dtype=float)
        predictions = np.array(f['predictions'], dtype=int)
        answers     = np.array(f['answers'],     dtype=int)

    batch_size, max_len = outputs.shape
    dim1 = int(np.floor(np.sqrt(max_len)))
    dim2 = int(np.ceil(float(max_len) / dim1))

    ip = np.concatenate((inputs, np.zeros((batch_size, dim1 * dim2 - max_len))), axis = 1)
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
      if answers[b] != 0:
        if args.in_context_only and answers[b] not in ip[b]:
          continue

        if args.incorrect_only and answers[b] == predictions[b]:
          continue

        image_file = os.path.abspath(filename) + '.ex{}.png'.format(str(b).zfill(3))
        if os.path.isfile(image_file):
          continue

        if args.in_context_only:
          word_to_prob = {}
          for w in range(ip.shape[1]):
            word = ip[b][w]
            if word != 0:
              if word not in word_to_prob:
                word_to_prob[word] = 0
              word_to_prob[word] += op[b][w]

          answer_rank = 1
          for word in word_to_prob:
            if word != answers[b] and word_to_prob[word] > word_to_prob[answers[b]]:
              answer_rank += 1

          if args.answer_rank > 0 and answer_rank != args.answer_rank:
            continue

          if debug_file:
            debug_file.write(u'=hyperlink("{}")\t{}\t{}\t{}\n'.format(
              image_file, answer_rank, idx2word[answers[b]], idx2word[predictions[b]]))
          
        analysis_result = ''
        if len(category_labels) > 0: # if there is analysis result for this file

          analysis_index = -1
          lookup = ' '.join([str(t) for t in inputs[b,:]])
          for i in range(len(contexts)):
            if contexts[i] in lookup:
              analysis_index = i
              break

          assert analysis_index >= 0, 'Example not found in analysis result: {}'.format(lookup)

          found_labels = []
          for category in category_labels:
            if category_labels[category][analysis_index] == 1:
              found_labels.append(category)

              analysis_category_count[category] += 1
              if answers[b] == predictions[b]:
                analysis_category_correct[category] += 1


          analysis_result = ' , '.join(found_labels)

        ax = plt.axes()
        sns.heatmap(op[b].reshape((dim1,dim2)), annot=labels[b].reshape((dim1,dim2)), fmt='', cmap='Blues', ax = ax)
        answer_indicator = r"$\bf{CORRECT}$" if answers[b] == predictions[b] else "Correct"
        answer_indicator = answer_indicator + ' (in context)' if answers[b] in ip[b] else answer_indicator

        title_text = '{} = {}, Prediction = {}'.format(answer_indicator, idx2word[answers[b]], idx2word[predictions[b]])
        if len(analysis_result) > 0:
            title_text += '\nExample Types:{{ {} }}'.format(analysis_result)
        ax.set_title(title_text)

        # plt.show()
        savefig(image_file, bbox_inches='tight', dpi = 100)
        plt.clf()

  for cat in category_labels:
    print('Category: {}, Accuracy = {}% ({} out of {})'.format(
      cat, analysis_category_correct[cat] * 100.0 / analysis_category_count[cat], analysis_category_correct[cat], analysis_category_count[cat]))

  if len(args.out_debug_file):
    debug_file.close()

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
