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
        self.softmax = nn.Softmax()
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

        doc_states, _ = self.doc_rnn(inp) # seqlen x bsz x 2*rnn_size
        query_states, _ = self.query_rnn(inp) # seqlen x bsz x 2*rnn_size
        assert query_states.size(0) == seqlen
        # take last forward state and first bwd state
        query_rep = torch.cat([query_states[seqlen-1, :, :self.rnn_size],
                               query_states[0, :, self.rnn_size:]], 1) # bsz x 2*rnn_size

        if self.topdrop and self.dropout.p > 0:
            query_rep = self.dropout(query_rep)

        # bsz x seqlen x 2*rnn_size * bsz x 2*rnn_size x 1 -> bsz x seqlen x 1 -> bsz x seqlen
        scores = torch.bmm(doc_states.transpose(0, 1), query_rep.unsqueeze(2)).squeeze(2)
        # TODO: punctuation shit
        doc_mt_scores, query_mt_scores = None, None
        if self.mt_loss == "idx-loss":
            doc_mt_scores, query_mt_scores = self.get_step_scores(doc_states, query_states)
        elif self.mt_loss == "antecedent-loss":
            doc_mt_scores, query_mt_scores = self.get_ant_scores(doc_states, query_states)

        return self.softmax(scores), doc_mt_scores, query_mt_scores

    def get_states_for_step(self, states):
        """
        gets the states we want for doing multiclass pred @ time t
        args:
          states - seqlen x bsz x 2*rnn_size
        returns:
          seqlen*bsz x something
        """
        seqlen, bsz, drnn_sz = states.size()

        if not hasattr(self, "dummy"):
            self.dummy = states.data.new(1, drnn_sz/2).zero_()
        dummy = self.dummy

        if self.mt_step_mode == "exact":
            nustates = states.view(-1, drnn_sz) # seqlen*bsz x 2*rnn_size
        elif self.mt_step_mode == "before-after":
            dummyvar = Variable(dummy.expand(bsz, drnn_sz/2))
            # prepend zeros to front, giving seqlen*bsz x rnn_size
            fwds = torch.cat([dummyvar, states.view(-1, drnn_sz)[:-bsz, :drnn_sz/2]], 0)
            # append zeros to back, giving seqlen*bsz x rnn_size
            bwds = torch.cat([states.view(-1, drnn_sz)[bsz:, drnn_sz/2:], dummyvar], 0)
            nustates = torch.cat([fwds, bwds], 1) # seqlen*bsz x 2*rnn_size
        elif self.mt_step_mode == "before": # just before
            dummyvar = Variable(dummy.expand(bsz, drnn_sz/2))
            # prepend zeros to front, giving seqlen*bsz x rnn_size
            nustates = torch.cat([dummyvar, states.view(-1, drnn_sz)[:-bsz, :drnn_sz/2]], 0)
        else:
            assert False, "%s not a thing" % self.mt_step_mode
        return nustates


    def get_step_scores(self, doc_states, query_states):
        """
        doc_states - seqlen x bsz x 2*rnn_size
        query_states - seqlen x bsz x 2*rnn_size
        """
        states_for_step = self.get_states_for_step(doc_states)
        doc_mt_preds = self.doc_mt_lin(states_for_step) # seqlen*bsz x nclasses
        if self.query_mt:
            states_for_qstep = self.get_states_for_step(query_states)
            query_mt_preds = self.query_mt_lin(states_for_qstep)
        else:
            query_mt_preds = None
        return doc_mt_preds, query_mt_preds



def attn_sum_loss(batch, scores):
    """
    scores - bsz x seqlen
    answers - bsz
    """
    bsz, seqlen = scores.size()
    mask = batch["answers"].unsqueeze(1).expand(bsz, seqlen).eq(batch["words"].t())
    marg_log_prob_sum = (scores * Variable(mask.float())).sum(1).log().sum()
    return -marg_log_prob_sum


xent = nn.CrossEntropyLoss(ignore_index=0)

def multitask_loss1(batch, doc_mt_scores, query_mt_scores):
    """
    doc_mt_scores - seqlen*bsz x nclasses
    query_mt_scores - seqlen*bsz x nclasses
    """
    mt1_doc_labels = batch["mt1_doc_labels"] # seqlen*bsz, w/ 0 where we want to ignore
    loss = xent(doc_mt_scores, mt1_doc_labels)
    if query_mt_scores is not None:
        mt1_query_labels = batch["mt1_query_labels"] # seqlen*bsz, w/ 0 where we want to ignore
        loss = loss + xent(query_mt_scores, mt1_query_labels)
    return loss


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
