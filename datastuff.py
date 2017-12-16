"""
data crud
"""

import torch
import h5py

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


def make_mt1_targ_idxs(batch, max_entities, max_mentions):
    words = batch["words"]
    ner = batch["feats"][:, :, 1]
    bsz, seqlen = words.size()
    targ_idxs = torch.zeros(seqlen, bsz)
    for b in xrange(bsz):
        ments = 0
        uniq_ents = {}
        for i in xrange(seqlen):
            if ments <= max_mentions and ner[i][b] == 2: # tagged PERSON
                if words[i][b] in uniq_ents:
                    targ_idxs[i][b] = uniq_ents[words[i][b]]
                    ments += 1
                elif len(uniq_ents) < max_entities:
                    uniq_ents[words[i][b]] = len(uniq_ents)+1 # b/c 0 is ignored
                    targ_idxs[i][b] = uniq_ents[words[i][b]]
                    ments += 1
    return targ_idxs

class DataStuff(object):

    def __init__(self, args):
        h5dat = h5py.File(args.datafile)
        # h5 keys are:
        # ner_vocab_size, post_vocab_size, punctuations, sent_vocab_size, spee_vocab_size,
        # stopwords, test_data, test_extr, test_location, test_ner, test_post, test_sentence,
        # test_sid, test_speech, train_data, train_extr, train_location, train_ner, train_post,
        # train_sentence, train_sid, train_speech, valid_data, valid_extr, valid_location,
        # valid_ner, valid_post, valid_sentence, valid_sid, valid_speech, vocab_size,
        # word_embeddings'
        dat = {}
        for key in h5dat.keys():
            if key.startswith("train") or key.startswith("valid"):
                dat[key] = torch.from_numpy(h5dat[key][:])

        words_new2old, words_old2new = reduce_vocab([dat["train_data"], dat["valid_data"]])
        print "new vocab size:", len(words_new2old)
        self.words_new2old, self.words_old2new = words_new2old, words_old2new

        # replace words w/ new vocab
        for key in ['train_data', 'valid_data']:
            for i in xrange(dat[key].size(0)):
                dat[key][i] = words_old2new[dat[key][i]]

        # hold on to word embs for a bit
        self.word_embs = torch.from_numpy(h5dat["word_embeddings"][:])

        # we also want to do this for speaker_ids if we have them
        if args.speaker_feats:
            sid_new2old, sid_old2new = reduce_vocab([dat["train_sid"], dat["valid_sid"]])
            self.sid_new2old, self.sid_old2new = sid_new2old, sid_old2new

            # replace
            for key in ['train_sid', 'valid_sid']:
                for i in xrange(dat[key].size(0)):
                    dat[key][i] = sid_old2new[dat[key][i]]

        # make offsets 0-indexed
        for key in ['train_location', 'valid_location']:
            dat[key][:, 0].sub_(1) # first column is offsets

        self.ntrain = dat["train_location"].size(0)
        self.nvalid = dat["valid_location"].size(0)
        self.dat = dat

        # we need to increment feature indexes so we don't overlap
        pos_voc_size = h5dat['post_vocab_size'][:][0]
        self.dat["train_ner"].add_(pos_voc_size+1)
        self.dat["valid_ner"].add_(pos_voc_size+1)
        ner_voc_size = h5dat['ner_vocab_size'][:][0]
        self.dat["train_sentence"].add_(pos_voc_size+ner_voc_size+1)
        self.dat["valid_sentence"].add_(pos_voc_size+ner_voc_size+1)
        sent_voc_size = h5dat['sent_vocab_size'][:][0]
        self.feat_voc_size = pos_voc_size + ner_voc_size + sent_voc_size

        spee_voc_size = h5dat['spee_vocab_size'][:][0]
        self.dat["train_sid"].add_(spee_voc_size+1)
        self.dat["valid_sid"].add_(spee_voc_size+1)
        self.spee_feat_foc_size = max(self.dat["train_sid"].max(), self.dat["valid_sid"].max())

        self.extra_size = dat["train_extr"].size(1)
        self.mt_loss = args.mt_loss
        if self.mt_loss == "idx-loss":
            self.cache = {}

        self.word_ctx = torch.LongTensor()
        self.answers = torch.LongTensor()
        self.linenos = torch.LongTensor()
        self.feats = torch.LongTensor()
        self.extr = torch.Tensor()
        self.spee_feats = torch.LongTensor()
        self.choices = torch.Tensor()

        h5dat.close()


    def load_data(self, batch_idx, args, train=True):
        """
        dat is a dict w/ all the data stuff
        batch_idx is the idx of first thing in the batch
        """
        dat = self.dat
        pfx = "train" if train else "valid"
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

        if args.use_choices:
            self.choices.resize_(bsz, 10).zero_()

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

            if args.use_choices:
                self.choices[b].copy_(dat["%s_choices" % pfx][ex_idx])

        batch = {"words": self.word_ctx, "answers": self.answers}
        if args.std_feats:
            batch["feats"] = self.feats
            batch["extr"] = self.extr
        if args.speaker_feats:
            batch["spee_feats"] = self.spee_feats
        if args.use_choices:
            batch["choices"] = self.choices

        if self.mt_loss == "idx-loss":
            if batch_idx not in self.cache:
                targs = make_mt1_targ_idxs(batch, args.max_entities, args.max_mentions)
                self.cache[batch_idx] = targs
            batch["mt1_targs"] = self.cache[batch_idx]

        return batch

    def del_word_embs(self):
        del self.word_embs
