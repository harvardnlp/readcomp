"""
pytorch reimplementation of urrthing
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import h5py

class Reader(nn.Module):
    """
    attn sum style reader dealie
    """
    def __init__(self, word_embs, opt):
        super(Reader, self).__init__()
        self.wlut = nn.Embedding(opt.wordtypes, opt.emb_size)
        self.flut = nn.Embedding(opt.ftypes, opt.emb_size if opt.add_inp else opt.feat_size)
        self.splut = nn.Embedding(opt.sptypes, opt.emb_size if opt.add_inp else opt.sp_size)
        self.emb_size, self.rnn_size, self.add_inp = opt.emb_size, opt.rnn_size, opt.add_inp
        self.std_feats, self.speaker_feats = opt.std_feats, opt.speaker_feats
        insize = opt.emb_size
        if opt.std_feats and not opt.add_inp:
            insize += opt.feat_size + opt.extra_size
        if opt.speaker_feats and not opt.add_inp:
            insize += opt.sp_size
        self.doc_rnn = nn.GRU(insize, opt.rnn_size, opt.layers, bidirectional=True)
        self.query_rnn = nn.GRU(insize, opt.rnn_size, opt.layers)
        self.drop = nn.Dropout(opt.dropout)
        if opt.add_inp:
            self.extr_lin = nn.Linear(opt.extra_size, opt.emb_size)
        else:
            self.extr_mul = nn.Parameter(torch.Tensor(1, 1, opt.extra_size))
        self.inp_activ = nn.ReLU() if opt.relu else nn.Tanh()
        self.softmax = nn.SoftMax()
        self.init_weights(word_embs)

    def init_weights(self, word_embs):
        assert False

    def forward(self, batch):
        """
        returns bsz x seqlen scores
        """
        seqlen, bsz = batch["words"].size()
        wembs = self.wlut(batch["words"]) # seqlen x bsz -> seqlen x bsz x emb_size
        if self.std_feats:
            # seqlen x bsz x 3 -> seqlen x bsz*3 x emb_size -> seqlen x bsz x 3 x emb_size
            fembs = self.wlut(batch["feats"].view(seqlen, -1)).view(seqlen, bsz, -1)
        if self.speaker_feats:
            # seqlen x bsz x 2 -> seqlen x bsz*2 x emb_size -> seqlen x bsz x 2 x emb_size
            sembs = self.splut(batch["spee_feats"].view(seqlen, -1)).view(
              seqlen, bsz, -1, self.emb_size)
        inp = wembs
        if self.add_inp: # mlp the input
            if self.std_feats:
                ex_size = self.extra_lin(batch["extr"].size(2))
                inp = (inp + fembs.sum(2)
                           + self.extr_lin(batch["extr"].view(-1, ex_size)).view(seqlen, bsz, -1))
            if self.speaker_feats:
                inp = inp + sembs.sum(2)
            if self.std_feats or self.speaker_feats:
                inp = self.inp_activ(inp)
        else: # concatenate everything
            things_to_cat = [inp]
            if self.std_feats:
                things_to_cat.append(fembs.view(seqlen, bsz, -1))
                things_to_cat.append(batch["extr"] * self.extr_mul.expand_as(batch["extr"]))
            if self.speaker_feats:
                things_to_cat.append(sembs.view(seqlen, bsz, -1))
            if len(things_to_cat) > 1:
                inp = torch.cat(things_to_cat, 2) # seqlen x bsz x sum (all the stuff)

        if self.dropout.p > 0:
            inp = self.dropout(inp)

        doc_outs, _ = self.doc_rnn(inp) # seqlen x bsz x 2*rnn_size
        query_outs, _ = self.query_rnn(inp) # seqlen x bsz x 2*rnn_size
        assert query_outs.size(0) == seqlen
        query_rep = torch.cat([query_outs[seqlen-1, :, :self.rnn_size],
                               query_outs[0, :, self.rnn_size:]], 1) # bsz x 2*rnn_size

        if self.topdrop and self.dropout.p > 0:
            query_rep = self.dropout(query_rep)

        # bsz x seqlen x 2*rnn_size * bsz x 2*rnn_size x 1 -> bsz x seqlen x 1 -> bsz x seqlen
        scores = torch.bmm(doc_outs.transpose(0, 1), query_rep.unsqueeze(2)).squeeze(2)
        # TODO: punctuation shit
        return self.softmax(scores)


def attn_sum_loss(batch, scores):
    """
    scores - bsz x seqlen
    answers - bsz
    """
    bsz, seqlen = scores.size()
    mask = batch["answers"].unsqueeze(1).expand(bsz, seqlen).eq(batch["words"].t())
    marg_log_prob_sum = (scores * Variable(mask.float())).sum(1).log().sum()
    return -marg_log_prob_sum


h5dat = h5py.File(args.datafile)
# keys are:
# ner_vocab_size, post_vocab_size, punctuations, sent_vocab_size, spee_vocab_size,
# stopwords, test_data, test_extr, test_location, test_ner, test_post, test_sentence,
# test_sid, test_speech, train_data, train_extr, train_location, train_ner, train_post,
# train_sentence, train_sid, train_speech, valid_data, valid_extr, valid_location,
# valid_ner, valid_post, valid_sentence, valid_sid, valid_speech, vocab_size,
# word_embeddings'
datstuff = {}
for key in h5dat.keys():
    if key.startswith("train") or key.startswith("valid"):
        datstuff[key] = torch.from_numpy(h5dat[key][:])

vocab_map = [0] # maps new vocab indices to old indices
rev_vocab_map = {0:0}
with open(args.reduced_vocab_mapper_fi) as f:
    for line in f:
        nu, orig = line.strip().split()
        nu, orig = int(nu), int(orig)
        nu += 1 # we started w/ a dummy idx
        assert nu == len(vocab_map)
        vocab_map.append(orig)
        rev_vocab_map[orig] = nu

# replace words w/ new vocab
for key in ['train_data', 'valid_data']:
    for i in xrange(datstuff[key].size(0)):
        datstuff[key][i] = rev_vocab_map[datstuff[key][i]]

# make offsets 0-indexed
for key in ['train_location', 'valid_location']:
