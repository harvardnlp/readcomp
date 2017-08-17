import numpy as np
import codecs
import os
import nltk

UNKNOWN = '<unk>'
GLOVE_DIM = 100
SEPARATOR = '<sep>'
end_words  = { "?", "??", "???", "!", "!!", "!!!", ".", "?!", "!?" }

def print_msg(message, verbose_level, args_verbose_level):
  if args_verbose_level >= verbose_level:
    print message


class Dictionary(object):
  def __init__(self):
    self.word2idx = {}
    self.word2count = {} # for NCE if training LM
    self.idx2word = []

    self.post2idx = {} # pos tags
    self.punc2idx = {} # punctuations
    self.stop2idx = {} # stop words


  def add_word(self, word):
    if word not in self.word2idx:
      self.idx2word.append(word)
      self.word2idx[word] = len(self.idx2word) - 1 # subtract 1 to make <sep> token index 0
      self.word2count[word] = 0 # set to 0 since a word in vocab may not appear in training data

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

    with codecs.open(file_prefix + '.pos.vocab', 'r', encoding='utf8') as posf:
      for line in posf:
        parts = line.split()
        self.post2idx[parts[0]] = int(parts[1])


class Corpus(object):
  def __init__(self, args_verbose_level, vocab_file, glove_file, glove_size, punc_file, stop_word_file, extra_vocab_file, context_target_separator, answer_identifier):
    self.args_verbose_level = args_verbose_level
    self.context_target_separator = context_target_separator # special separator token to identify context and target
    self.answer_identifier = answer_identifier
    self.dictify(vocab_file, glove_file, glove_size, punc_file, stop_word_file, extra_vocab_file)


  def dictify(self, vocab_file, glove_file, glove_size, punc_file, stop_word_file, extra_vocab_file):
    self.dictionary = Dictionary()

    if vocab_file != None or glove_file != None:
      self.dictionary.add_word(SEPARATOR) # map to 0 for masked rnn
      self.dictionary.add_word(UNKNOWN)
      if vocab_file != None:
        with codecs.open(vocab_file, 'r', encoding="utf-8") as f:
          for line in f:
            if line.strip():
              self.dictionary.add_word(line.strip())
      else:
        print_msg('Loading GLOVE ...', 1, self.args_verbose_level)
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
        print_msg('Done ...', 1, self.args_verbose_level)

    if len(self.dictionary) > 0:
      self.punctuations = []
      self.stopwords = []

      with codecs.open(punc_file, 'r', encoding="utf-8") as f:
        print_msg('Loading punctuations ...', 1, self.args_verbose_level)
        for line in f:
          punc = line.strip()
          if punc:
            self.punctuations.append(self.dictionary.add_word(punc))
            if punc not in self.dictionary.punc2idx:
              self.dictionary.punc2idx[punc] = len(self.dictionary.punc2idx) + 1

      with codecs.open(stop_word_file, 'r', encoding="utf-8") as f:
        print_msg('Loading stop words ...', 1, self.args_verbose_level)
        for line in f:
          sw = line.strip()
          if sw:
            self.stopwords.append(self.dictionary.add_word(sw))
            if sw not in self.dictionary.stop2idx:
              self.dictionary.stop2idx[sw] = len(self.dictionary.stop2idx) + 1

      if extra_vocab_file:
        with codecs.open(extra_vocab_file, 'r', encoding="utf-8") as f:
          print_msg('Loading extra vocab ...', 1, self.args_verbose_level)
          for line in f:
            self.dictionary.add_word(line.strip())

      print 'Vocab size = {}'.format(len(self.dictionary), 1, self.args_verbose_level)


  def load_vocab(self, vocab_file_prefix):
    print_msg('Loading vocab...', 1, self.args_verbose_level)
    self.dictionary.read_from_file(vocab_file_prefix)


  def load(self, path, train, valid, test, control, analysis):
    print_msg('Loading train data ...', 1, self.args_verbose_level)
    self.train   = self.tokenize(os.path.join(path, train),   training = True)
    print_msg('Loading validation data...', 1, self.args_verbose_level)
    self.valid   = self.tokenize(os.path.join(path, valid),   training = True)
    print_msg('Loading test data...', 1, self.args_verbose_level)
    self.test    = self.tokenize(os.path.join(path, test),    training = False)
    print_msg('Loading control data...', 1, self.args_verbose_level)
    self.control = self.tokenize(os.path.join(path, control), training = False)
    print_msg('Loading analysis data...', 1, self.args_verbose_level)
    self.analysis = self.tokenize(os.path.join(path, analysis), training = False)

    print_msg('\nTraining Data Statistics:\n', 1, self.args_verbose_level)
    train_context_length = self.train['location'][:,1]
    train_context_length = train_context_length[train_context_length > 0]
    print_msg('Context Length: max = {}, min = {}, average = {}, std = {}'.format(
      np.max(train_context_length), np.min(train_context_length), np.mean(train_context_length), np.std(train_context_length)), 1, self.args_verbose_level)

    print_msg('POS Size: {}'.format(len(self.dictionary.post2idx)), 1, self.args_verbose_level)


  def save(self, file_prefix):
    self.dictionary.write_to_file(file_prefix)


  def tokenize(self, path, training):
    assert os.path.exists(path)
    
    data = { 
      'data': [], # token ids for each word in the corpus 
      'post': [], # pos tags 
      'extr': [], # extra features, such as frequency of token in the context, whether previous bi-gram of token match with that of the answer etc...
      'offsets': [], # offset locations for each line in the final 1-d data array 
      'context_length': [], # count of words in the context
    }

    self.tokenize_file(path, data, training)

    sorted_data = { 'data': data['data'], 'post': data['post'], 'extr': data['extr'] }

    loc = np.array([np.array(data['offsets']), np.array(data['context_length'])]).T
    loc = loc[np.argsort(-loc[:,1])] # sort by context length in descending order
    sorted_data['location'] = loc
    return sorted_data

 
  # update the ids, offsets, word counts, line counts
  def tokenize_file(self, file, data, training):
    num_lines_in_file = 0
    puncstop_answer_count = 0

    with codecs.open(file, 'r', encoding='utf8') as f:
      for line in f:
        num_lines_in_file += 1
        words = line.split()
        
        if self.context_target_separator:
          if num_lines_in_file == 1:
            print_msg('INFO: Using context-query-answer separator token = {}'.format(self.context_target_separator), 1, self.args_verbose_level)

          sep = words.index(self.context_target_separator) - 1
          if sep <= 2:
            print_msg('INFO: SKIPPING... Context should contain at least 2 tokens, line = {}'.format(line), 2, self.args_verbose_level)
            continue

          words.pop(sep + 1) # remove separator
          target_answer_separator_index = words.index(self.context_target_separator)
          if target_answer_separator_index <= 0:
            print_msg('INFO: SKIPPING... Target-Answer separator not found, line = {}'.format(line), 2, self.args_verbose_level)
            continue
          words.pop(target_answer_separator_index)

        num_words = len(words)
        pos_tags = [t[1] for t in nltk.pos_tag(words)]

        answer = words[num_words - 1]
        self.dictionary.add_word(answer) # make sure the answer is in vocab

        if answer in self.dictionary.punc2idx or answer in self.dictionary.stop2idx:
          puncstop_answer_count += 1
          print_msg('INFO: SKIPPING... Target-Answer is either a stop word or punctuation, line = {}'.format(line), 2, self.args_verbose_level)
          continue

        if re.search('[a-z]', answer) == None:
          print_msg('INFO: SKIPPING... Target-Answer is not valid, line = {}'.format(line), 2, self.args_verbose_level)
          continue

        if training:
          # make sure answer is part of context (for computing loss & gradients during training)
          found_answer = False
          for i in range(0, num_words - 1):
            if answer == words[i]:
              found_answer = True
          if not found_answer:
            print_msg('INFO: SKIPPING... Target answer not found in context', 2, self.args_verbose_level)
            continue

        data['offsets'].append(len(data['data']) + 1)
        data['context_length'].append(num_words - 1)

        words = [word if word in self.dictionary.word2idx else UNKNOWN for word in words]

        extr_word_freq = {}
        for i in range(len(words)):
          word = words[i]

          if word not in extr_word_freq:
            extr_word_freq[word] = 0

          # only count within context for non-punctuation and non-stopword tokens
          if i < num_words - 1 and word not in self.dictionary.punc2idx and word not in self.dictionary.stop2idx:
            extr_word_freq[word] += 1

          data['data'].append(self.dictionary.word2idx[word])

          pos_tag = pos_tags[i]
          data['post'].append(self.dictionary.add_pos_tag(pos_tag))

          self.dictionary.update_count(word)

        
        for i in range(len(words)):
          word = words[i]
          extra_features = []

          freq = float(extr_word_freq[word]) / len(words)
          bigram_match = 0
          if i < num_words - 1:
            if self.answer_identifier: # if location of answer is identified in the query (e.g. for CNN dataset) 
              if num_lines_in_file == 1 and i == 0:
                print_msg('INFO: Using answer identifier token = {}'.format(self.answer_identifier), 1, self.args_verbose_level)
              answer_index = words.index(self.answer_identifier)

              # make sure the previous and next ngrams of the token are actually in the context
              # and vice versa for the target answer 
              if i > 1 and answer_index > 1 and i != answer_index and words[i - 1] == words[answer_index - 1]:
                bigram_match = 0.5
              elif i < num_words - 1 and answer_index < num_words - 1 and i != answer_index and words[i + 1] == words[answer_index + 1]:
                bigram_match = 0.5

              if i > 2 and answer_index > 2 and i != answer_index and words[i - 2] == words[answer_index - 2] and words[i - 1] == words[answer_index - 1]:
                bigram_match = 1
              elif i < num_words - 2 and answer_index < num_words - 2 and i != answer_index and words[i + 1] == words[answer_index + 1] and words[i + 2] == words[answer_index + 2]:
                bigram_match = 1

              if i > 3 and answer_index > 3 and i != answer_index and words[i - 3] == words[answer_index - 3] and words[i - 2] == words[answer_index - 2] and words[i - 1] == words[answer_index - 1]:
                bigram_match = 1.5
              elif i < num_words - 3 and answer_index < num_words - 3 and i != answer_index and words[i + 1] == words[answer_index + 1] and words[i + 2] == words[answer_index + 2]  and words[i + 3] == words[answer_index + 3]:
                bigram_match = 1.5

            else: # if not assume the location is at the end (e.g. LAMBADA)
              if i > 1 and words[i - 1] == words[num_words - 2]:
                bigram_match = 0.5

              if i > 2 and words[i - 2] == words[num_words - 3] and words[i - 1] == words[num_words - 2]:
                bigram_match = 1

              if i > 3 and words[i - 3] == words[num_words - 4] and words[i - 2] == words[num_words - 3] and words[i - 1] == words[num_words - 2]:
                bigram_match = 1.5

          extra_features.append(freq)
          extra_features.append(bigram_match)

          data['extr'].append(np.array(extra_features))

        print_msg('Processed {} lines'.format(num_lines_in_file), 3, self.args_verbose_level)

    print_msg('\nCount of cases where answer is a punctuation symbol or stop word: ' + str(puncstop_answer_count), 1, self.args_verbose_level)
