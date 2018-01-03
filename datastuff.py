"""
data crud
"""

import torch
import h5py
from datamodel import Dictionary
import numpy as np

def reduce_vocab(word_vecs):
    """
    just takes words already used, but reserves 0
    """
    uniques = set()
    for wv in word_vecs:
        uniques.update(wv)
    new2old = [0] # always map 0 to itself
    if 0 in uniques:
        uniques.remove(0)
    new2old.extend(sorted(uniques))
    old2new = dict((w, i) for i, w in enumerate(new2old))
    return new2old, old2new


def make_mt1_targ_idxs(batch, max_entities, max_mentions, per_idx):
    words = batch["words"]
    ner = batch["feats"][:, :, 1]
    seqlen, bsz = words.size()
    targ_idxs = torch.LongTensor(seqlen, bsz).zero_()

    for b in xrange(bsz):
        ments = 0
        uniq_ents = {}
        for i in xrange(seqlen):
            if ments <= max_mentions and ner[i][b] == per_idx: # tagged PERSON
                if words[i][b] in uniq_ents:
                    targ_idxs[i][b] = uniq_ents[words[i][b]]
                    ments += 1
                elif len(uniq_ents) < max_entities:
                    uniq_ents[words[i][b]] = len(uniq_ents)+1 # b/c 0 is ignored
                    targ_idxs[i][b] = uniq_ents[words[i][b]]
                    ments += 1
    return targ_idxs


def make_mt2_targs(batch, max_entities, max_mentions, per_idx):
    words = batch["words"]
    ner = batch["feats"][:, :, 1]
    seqlen, bsz = words.size()
    rep_ents = torch.zeros(bsz, seqlen) # entities that have been repeated

    for b in xrange(bsz):
        ments = 0
        uniq_ents = set()
        for i in xrange(seqlen):
            if ments <= max_mentions and ner[i][b] == per_idx: # tagged PERSON
                if words[i][b] in uniq_ents:
                    rep_ents[b][i] = 1
                    ments += 1
                elif len(uniq_ents) < max_entities:
                    uniq_ents.add(words[i][b]) # it's first so don't add it

    return rep_ents

class DataStuff(object):

    def __init__(self, args):
        h5dat = h5py.File(args.datafile, 'r')
        # h5 keys are:
        # ner_vocab_size, post_vocab_size, punctuations, sent_vocab_size, spee_vocab_size,
        # stopwords, test_data, test_extr, test_location, test_ner, test_post, test_sentence,
        # test_sid, test_speech, train_data, train_extr, train_location, train_ner, train_post,
        # train_sentence, train_sid, train_speech, valid_data, valid_extr, valid_location,
        # valid_ner, valid_post, valid_sentence, valid_sid, valid_speech, vocab_size,
        # word_embeddings'
        dat = {}
        for key in h5dat.keys():
            if key.startswith("train") or key.startswith("valid") or key.startswith("test"):
                dat[key] = torch.from_numpy(h5dat[key][:])

        words_new2old, words_old2new = reduce_vocab([dat["train_data"], dat["valid_data"],
                                                     dat["test_data"]])
        print "new vocab size:", len(words_new2old)
        self.words_new2old, self.words_old2new = words_new2old, words_old2new

        # replace words w/ new vocab
        for key in ['train_data', 'valid_data', 'test_data']:
            for i in xrange(dat[key].size(0)):
                dat[key][i] = words_old2new[dat[key][i]]

        if args.use_choices or args.use_test_choices:
            for key in ['train_choices', 'valid_choices', 'test_choices']:
                vec = dat[key].view(-1)
                for i in xrange(vec.size(0)):
                    vec[i] = words_old2new[vec[i]]

        if args.use_qidx:
            self.query_idx = self.words_old2new[args.query_idx]

        # hold on to word embs for a bit
        self.word_embs = torch.from_numpy(h5dat["word_embeddings"][:])

        # we also want to do this for speaker_ids if we have them
        if args.speaker_feats:
            sid_new2old, sid_old2new = reduce_vocab([dat["train_sid"], dat["valid_sid"],
                                                     dat["test_sid"]])
            self.sid_new2old, self.sid_old2new = sid_new2old, sid_old2new

            # replace
            for key in ['train_sid', 'valid_sid', 'test_sid']:
                for i in xrange(dat[key].size(0)):
                    dat[key][i] = sid_old2new[dat[key][i]]

        # make offsets 0-indexed
        for key in ['train_location', 'valid_location', 'test_location']:
            dat[key][:, 0].sub_(1) # first column is offsets

        self.ntrain = dat["train_location"].size(0)
        self.nvalid = dat["valid_location"].size(0)
        self.ntest  = dat["test_location"].size(0)
        self.dat = dat

        # we need to increment feature indexes so we don't overlap
        pos_voc_size = h5dat['post_vocab_size'][:][0]+1
        self.dat["train_ner"].add_(pos_voc_size)
        self.dat["valid_ner"].add_(pos_voc_size)
        self.dat["test_ner" ].add_(pos_voc_size)
        self.per_idx = 2 + pos_voc_size # 2 is PERSON
        ner_voc_size = h5dat['ner_vocab_size'][:][0]+1
        self.dat["train_sentence"].add_(pos_voc_size+ner_voc_size)
        self.dat["valid_sentence"].add_(pos_voc_size+ner_voc_size)
        self.dat["test_sentence" ].add_(pos_voc_size+ner_voc_size)
        #sent_voc_size = h5dat['sent_vocab_size'][:][0]+1
        self.feat_voc_size = max(self.dat["train_sentence"].max(),
                                 self.dat["valid_sentence"].max(),
                                 self.dat["test_sentence" ].max())+1

        spee_voc_size = h5dat['spee_vocab_size'][:][0]+1
        self.dat["train_sid"].add_(spee_voc_size)
        self.dat["valid_sid"].add_(spee_voc_size)
        self.dat["test_sid" ].add_(spee_voc_size)
        self.spee_feat_foc_size = max(self.dat["train_sid"].max(), self.dat["valid_sid"].max(), self.dat["test_sid"].max())+1

        self.extra_size = dat["train_extr"].size(1)
        self.mt_loss = args.mt_loss
        if self.mt_loss != "":
            self.cache = {}

        self.word_ctx = torch.LongTensor()
        self.answers = torch.LongTensor()
        self.linenos = torch.LongTensor()
        self.feats = torch.LongTensor()
        self.extr = torch.Tensor()
        self.spee_feats = torch.LongTensor()
        self.use_qidx = args.use_qidx
        if self.use_qidx:
            self.query_pos = torch.LongTensor()
        if args.use_choices or args.use_test_choices:
            self.choicemask = torch.Tensor()

        h5dat.close()


    def load_data(self, batch_idx, args, data_mode='train'):
        """
        dat is a dict w/ all the data stuff
        batch_idx is the idx of first thing in the batch
        """
        dat = self.dat
        pfx = data_mode
        train = data_mode == 'train'
        loc = dat["%s_location" % pfx] # nexamples x 3
        bsz = min(args.bsz, loc.size(0)-batch_idx)
        max_ctx_len = min(args.maxseqlen, loc[batch_idx:batch_idx+bsz, 1].max())
        self.word_ctx.resize_(max_ctx_len, bsz).zero_()
        self.answers.resize_(bsz).zero_()
        self.linenos.resize_(bsz).zero_()

        if args.std_feats:
            self.feats.resize_(max_ctx_len, bsz, 3).zero_()
            self.extr.resize_(max_ctx_len, bsz, self.extra_size).zero_()
        if args.speaker_feats:
            self.spee_feats.resize_(max_ctx_len, bsz, 2).zero_()

        if args.use_choices or (args.use_test_choices and not train):
            self.choicemask.resize_(bsz, max_ctx_len).zero_()

        if self.use_qidx:
            self.query_pos.resize_(bsz).fill_(-1) # assuming these always go together

        for b in xrange(bsz):
            ex_idx = batch_idx + b
            offset, ctx_len, self.linenos[b] = loc[ex_idx]
            capped_len = min(args.maxseqlen, ctx_len)
            answer_idx = offset + ctx_len
            self.answers[b] = dat["%s_data" % pfx][answer_idx]

            self.word_ctx[-capped_len:, b].copy_(
                dat["%s_data" % pfx][answer_idx-capped_len:answer_idx])
            if args.std_feats:
                self.feats[-capped_len:, b, 0].copy_(
                    dat["%s_post" % pfx][answer_idx-capped_len:answer_idx])
                self.feats[-capped_len:, b, 1].copy_(
                    dat["%s_ner" % pfx][answer_idx-capped_len:answer_idx])
                self.feats[-capped_len:, b, 2].copy_(
                    dat["%s_sentence" % pfx][answer_idx-capped_len:answer_idx])
                self.extr[-capped_len:, b, :].copy_(
                    dat["%s_extr" % pfx][answer_idx-capped_len:answer_idx])
            if args.speaker_feats:
                self.spee_feats[-capped_len:, b, 0].copy_(
                    dat["%s_speech" % pfx][answer_idx-capped_len:answer_idx])
                self.spee_feats[-capped_len:, b, 1].copy_(
                    dat["%s_sid" % pfx][answer_idx-capped_len:answer_idx])

            if args.use_choices or (args.use_test_choices and not train):
                bchoices = set(dat["%s_choices" % pfx][ex_idx])
                [self.choicemask[b].__setitem__(jj, 1) for jj in xrange(max_ctx_len)
                 if self.word_ctx[jj, b] in bchoices]

            if self.use_qidx:
                qpos = torch.nonzero(self.word_ctx[:, b] == self.query_idx)[0][0]
                self.query_pos[b] = qpos*bsz + b

        # if args.use_choices:
        #     # get bsz x 2 tensor of idxs (need to transpose below to be right)
        #     poss = torch.nonzero(self.word_ctx.t() == self.query_idx)
        #     self.query_pos.copy_(poss[:, 1]) # 2nd col has nz col in transpose

        batch = {"words": self.word_ctx, "answers": self.answers}
        if args.std_feats:
            batch["feats"] = self.feats
            batch["extr"] = self.extr
        if args.speaker_feats:
            batch["spee_feats"] = self.spee_feats
        if args.use_choices or (args.use_test_choices and not train):
            batch["choicemask"] = self.choicemask
        if self.use_qidx:
            batch["qpos"] = self.query_pos

        if self.mt_loss == "idx-loss":
            if data_mode not in self.cache:
                self.cache[data_mode] = {}
            if batch_idx not in self.cache[data_mode]:
                targs = make_mt1_targ_idxs(batch, args.max_entities,
                                           args.max_mentions, self.per_idx)
                self.cache[data_mode][batch_idx] = targs
            batch["mt1_targs"] = self.cache[data_mode][batch_idx]
        elif self.mt_loss == "ant-loss":
            if data_mode not in self.cache:
                self.cache[data_mode] = {}
            if batch_idx not in self.cache[data_mode]:
                targs = make_mt2_targs(batch, args.max_entities,
                                       args.max_mentions, self.per_idx)
                self.cache[data_mode][batch_idx] = targs
            batch["mt2_targs"] = self.cache[data_mode][batch_idx]

        return batch


    def analyze_data(self, vocab_file_prefix, batch, preds, answers, mt_scores, anares = {}):
        d = Dictionary()
        d.read_from_file(vocab_file_prefix)
        context = batch['words'].data.cpu().numpy() # seqlen x bsz
        seqlen, batchsize = context.shape

        if len(anares) == 0:
            anares['is_person'] = { 'correct': 0, 'total': 0 }
            anares['is_speaker'] = { 'correct': 0, 'total': 0 }
            anares['mt_is_person'] = { 'correct': 0, 'total': 0 }
            anares['mt_is_speaker'] = { 'correct': 0, 'total': 0 }

        if self.mt_loss == "idx-loss":
            mt_answers = batch["mt1_targs"].data.cpu().numpy() # seqlen x batchsize
            mt_mask = mt_answers != 0
            mt_scores = mt_scores.data.cpu().numpy().reshape(seqlen, batchsize, -1)
            mt_preds = np.argmax(mt_scores, 2)
            mt_diff = mt_preds[mt_mask] - mt_answers[mt_mask]
            anares['mt_is_person']['total'] += np.sum(mt_mask)
            anares['mt_is_person']['correct'] += np.sum(mt_diff == 0)
        elif self.mt_loss == 'ant-loss':
            mt_answers = batch["mt2_targs"].data.cpu().numpy() # bsz x seqlen
            mt_scores = mt_scores.data.cpu().numpy() # bsz x seqlen x seqlen

        for b in range(batchsize):
            w = ' '.join([d.idx2word[self.words_new2old[int(t)]] for t in context[:,b]])
            a = d.idx2word[self.words_new2old[int(answers[b])]]
            p = d.idx2word[self.words_new2old[int(preds[b])]]

            if 'speaker' in a:
                anares['is_person']['total'] += 1
                anares['is_person']['correct'] += 1 if answers[b] == preds[b] else 0

                # check for conversation if there are quotes
                contains_speech = (" `` " in w and " '' " in w) or (" ` " in w and " ' " in w)
                if contains_speech:
                    anares['is_speaker']['total'] += 1
                    anares['is_speaker']['correct'] += 1 if answers[b] == preds[b] else 0

                    if self.mt_loss == "idx-loss":
                        b_mt_mask = mt_mask[:,b]
                        mt_diff = mt_preds[:,b][b_mt_mask] - mt_answers[:,b][b_mt_mask]
                        anares['mt_is_speaker']['total'] += np.sum(b_mt_mask)
                        anares['mt_is_speaker']['correct'] += np.sum(mt_diff == 0)
                
                if self.mt_loss == "ant-loss":
                    # mt_scores is bsz x seqlen x seqlen
                    for s1 in range(seqlen):
                        if mt_answers[b,s1] > 0: # if repeated entity
                            anares['mt_is_person']['total'] += 1
                            anares['mt_is_speaker']['total'] += 1 if contains_speech else 0
                            w2p = {}
                            max_w, max_p = 0, 0
                            for s2 in range(s1 - 1):
                                cw = context[s2,b]
                                if cw not in w2p:
                                    w2p[cw] = 0
                                w2p[cw] += mt_scores[b,s1,s2]
                                if max_p < w2p[cw]:
                                    max_w = cw
                                    max_p = w2p[cw]
                            if max_w == context[s1,b]:
                                anares['mt_is_person']['correct'] += 1
                                anares['mt_is_speaker']['correct'] += 1 if contains_speech else 0
                        





    def del_word_embs(self):
        del self.word_embs
