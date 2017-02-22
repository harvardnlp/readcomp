import os
import torch
import fnmatch

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, train, valid, test):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, train))
        self.valid = self.tokenize(os.path.join(path, valid))
        self.test = self.tokenize(os.path.join(path, test))
        print 'Vocab size = {}'.format(len(self.dictionary))

    def dictify(self, file):
        with open(file, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)
            return tokens

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        
        if os.path.isdir(path):
            tokens = 0
            for root, dir_names, file_names in os.walk(path):
                for file in fnmatch.filter(file_names, '*.txt'):
                    tokens += self.dictify(os.path.join(root,file))

            ids = torch.LongTensor(tokens)
            token = 0
            for root, dir_names, file_names in os.walk(path):
                for file in fnmatch.filter(file_names, '*.txt'):
                    # Tokenize file content
                    with open(os.path.join(root,file), 'r') as f:
                        for line in f:
                            words = line.split() + ['<eos>']
                            for word in words:
                                ids[token] = self.dictionary.word2idx[word]
                                token += 1
        else:
            # Add words to the dictionary
            tokens = self.dictify(path)

            # Tokenize file content
            with open(path, 'r') as f:
                ids = torch.LongTensor(tokens)
                token = 0
                for line in f:
                    words = line.split() + ['<eos>']
                    for word in words:
                        ids[token] = self.dictionary.word2idx[word]
                        token += 1

        return ids
