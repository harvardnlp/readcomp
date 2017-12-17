"""
pytorch reimplementation of urrthing
"""
import argparse
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import datastuff

class Reader(nn.Module):
    """
    attn sum style reader dealie
    """
    def __init__(self, word_embs, words_new2old, opt):
        super(Reader, self).__init__()
        self.wlut = nn.Embedding(opt.wordtypes, opt.emb_size)
        if opt.std_feats:
            self.flut = nn.Embedding(opt.ftypes, opt.emb_size if opt.add_inp else opt.feat_size)
        if opt.speaker_feats:
            self.splut = nn.Embedding(opt.sptypes, opt.emb_size if opt.add_inp else opt.sp_size)
        self.emb_size, self.rnn_size, self.add_inp = opt.emb_size, opt.rnn_size, opt.add_inp
        self.std_feats, self.speaker_feats = opt.std_feats, opt.speaker_feats
        insize = opt.emb_size
        if opt.std_feats and not opt.add_inp:
            insize += 3*opt.feat_size + opt.extra_size
        if opt.speaker_feats and not opt.add_inp:
            insize += 2*opt.sp_size
        self.doc_rnn = nn.GRU(insize, 2*opt.rnn_size, opt.layers, bidirectional=True)
        self.drop = nn.Dropout(opt.dropout)
        if opt.add_inp:
            self.extr_lin = nn.Linear(opt.extra_size, opt.emb_size)
        else:
            self.extr_mul = nn.Parameter(
                torch.Tensor(1, 1, opt.extra_size).uniform_(-opt.initrange, opt.initrange))
        self.inp_activ = nn.ReLU() if opt.relu else nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.initrange = opt.initrange
        self.mt_loss, self.mt_step_mode = opt.mt_loss, opt.mt_step_mode
        if self.mt_loss == "idx-loss":
            mt_in = 2*opt.rnn_size if self.mt_step_mode == "before" else 4*opt.rnn_size
            self.doc_mt_lin = nn.Linear(mt_in, opt.max_entities+1) # 0 is an ignore idx
        self.topdrop, self.mt_drop = opt.topdrop, opt.mt_drop
        self.init_weights(word_embs, words_new2old)


    def init_weights(self, word_embs, words_new2old):
        """
        (re)init weights
        """
        initrange = self.initrange
        luts = [self.wlut]
        if self.std_feats:
            luts.append(self.flut)
        if self.speaker_feats:
            luts.append(self.splut)
        for lut in luts:
            lut.weight.data.uniform_(-initrange, initrange)

        rnns = [self.doc_rnn]
        for rnn in rnns:
            for thing in rnn.parameters():
                thing.data.uniform_(-initrange, initrange)

        lins = []
        if self.add_inp:
            lins.append(self.extr_lin)
        if self.mt_loss == "idx-loss":
            lins.append(self.doc_mt_lin)
        for lin in lins:
            lin.weight.data.uniform_(-initrange, initrange)
            lin.bias.data.zero_()

        # do the word embeddings
        for i in xrange(len(words_new2old)):
            old_idx = words_new2old[i]
            if old_idx < word_embs.size(0):
                self.wlut.weight.data[i][:word_embs.size(1)].copy_(word_embs[old_idx])

    def forward(self, batch):
        """
        returns bsz x seqlen scores
        """
        seqlen, bsz = batch["words"].size()
        wembs = self.wlut(batch["words"]) # seqlen x bsz -> seqlen x bsz x emb_size
        if self.std_feats:
            # seqlen x bsz x 3 -> seqlen x bsz*3 x emb_size -> seqlen x bsz x 3 x emb_size
            fembs = self.flut(batch["feats"].view(seqlen, -1)).view(
                seqlen, bsz, -1, self.flut.embedding_dim)
        if self.speaker_feats:
            # seqlen x bsz x 2 -> seqlen x bsz*2 x emb_size -> seqlen x bsz x 2 x emb_size
            sembs = self.splut(batch["spee_feats"].view(seqlen, -1)).view(
                seqlen, bsz, -1, self.splut.embedding_dim)
        inp = wembs
        if self.add_inp: # mlp the input
            if self.std_feats:
                ex_size = batch["extr"].size(2)
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

        if self.drop.p > 0:
            inp = self.drop(inp)

        # view each state as [fwd_q, fwd_d, bwd_d, bwd_q]
        states, _ = self.doc_rnn(inp) # seqlen x bsz x 2*2*rnn_size
        doc_states = states[:, :, self.rnn_size:3*self.rnn_size]
        query_rep = torch.cat([states[seqlen-1, :, :self.rnn_size],
                               states[0, :, -self.rnn_size:]], 1) # bsz x 2*rnn_size

        if self.topdrop and self.drop.p > 0:
            doc_states = self.drop(doc_states)
            #query_rep = self.drop(query_rep)

        # bsz x seqlen x 2*rnn_size * bsz x 2*rnn_size x 1 -> bsz x seqlen x 1 -> bsz x seqlen
        scores = torch.bmm(doc_states.transpose(0, 1), query_rep.unsqueeze(2)).squeeze(2)
        # TODO: punctuation shit
        doc_mt_scores = None
        if self.mt_loss == "idx-loss":
            doc_mt_scores = self.get_step_scores(states)
        elif self.mt_loss == "antecedent-loss":
            doc_mt_scores = self.get_ant_scores(states)

        return self.softmax(scores), doc_mt_scores

    def get_states_for_step(self, states):
        """
        gets the states we want for doing multiclass pred @ time t
        args:
          states - seqlen x bsz x 2*2*rnn_size; view each state as [fwd_q, fwd_d, bwd_d, bwd_q]
        returns:
          seqlen*bsz x something
        """
        seqlen, bsz, drnn_sz = states.size()

        if not hasattr(self, "dummy"):
            self.dummy = states.data.new(1, drnn_sz/2).zero_()
        dummy = self.dummy

        if self.mt_step_mode == "exact":
            nustates = states.view(-1, drnn_sz) # seqlen*bsz x 2*2*rnn_size
        elif self.mt_step_mode == "before-after":
            dummyvar = Variable(dummy.expand(bsz, drnn_sz/2))
            # prepend zeros to front, giving seqlen*bsz x 2*rnn_size
            fwds = torch.cat([dummyvar, states.view(-1, drnn_sz)[:-bsz, :drnn_sz/2]], 0)
            # append zeros to back, giving seqlen*bsz x 2*rnn_size
            bwds = torch.cat([states.view(-1, drnn_sz)[bsz:, drnn_sz/2:], dummyvar], 0)
            nustates = torch.cat([fwds, bwds], 1) # seqlen*bsz x 2*2*rnn_size
        elif self.mt_step_mode == "before": # just before
            dummyvar = Variable(dummy.expand(bsz, drnn_sz/2))
            # prepend zeros to front, giving seqlen*bsz x 2*rnn_size
            nustates = torch.cat([dummyvar, states.view(-1, drnn_sz)[:-bsz, :drnn_sz/2]], 0)
        else:
            assert False, "%s not a thing" % self.mt_step_mode
        return nustates


    def get_step_scores(self, states):
        """
        doc_states - seqlen x bsz x 2*rnn_size
        """
        states_for_step = self.get_states_for_step(states)
        if self.mt_drop and self.drop.p > 0:
            states_for_step = self.drop(states_for_step)
        doc_mt_preds = self.doc_mt_lin(states_for_step) # seqlen*bsz x nclasses
        return doc_mt_preds


def get_ncorrect(batch, scores):
    """
    i'm just gonna brute force this
    scores - bsz x seqlen
    answers - bsz
    """
    bsz, seqlen = scores.size()
    words, answers = batch["words"].data, batch["answers"].data
    ncorrect = 0
    for b in xrange(bsz):
        word2prob = defaultdict(float)
        best, best_prob = -1, -float("inf")
        for i in xrange(seqlen):
            word2prob[words[i][b]] += scores.data[b][i]
            if word2prob[words[i][b]] > best_prob:
                best = words[i][b]
                best_prob = word2prob[words[i][b]]
        ncorrect += (best == answers[b])
    return ncorrect


def attn_sum_loss(batch, scores):
    """
    scores - bsz x seqlen
    answers - bsz
    """
    bsz, seqlen = scores.size()
    mask = batch["answers"].data.unsqueeze(1).expand(bsz, seqlen).eq(batch["words"].data.t())
    marg_log_prob_sum = (scores * Variable(mask.float())).sum(1).log().sum()
    return -marg_log_prob_sum


xent = nn.CrossEntropyLoss(ignore_index=0, size_average=False)

def multitask_loss1(batch, doc_mt_scores):
    """
    doc_mt_scores - seqlen*bsz x nclasses
    """
    mt1_targs = batch["mt1_targs"] # seqlen x bsz, w/ 0 where we want to ignore
    loss = xent(doc_mt_scores, mt1_targs.view(-1))
    return loss


parser = argparse.ArgumentParser(description='')
parser.add_argument('-datafile', type=str, default='', help='')
parser.add_argument('-bsz', type=int, default=64, help='')
parser.add_argument('-maxseqlen', type=int, default=1024, help='')
parser.add_argument('-save', type=str, default='', help='path to save the final model')
parser.add_argument('-load', type=str, default='', help='path to saved model')

parser.add_argument('-std_feats', action='store_true', help='')
parser.add_argument('-speaker_feats', action='store_true', help='')
parser.add_argument('-use_choices', action='store_true', help='')
parser.add_argument('-mt_loss', type=str, default='',
                    choices=["", "idx-loss", "antecedent-loss"], help='')
parser.add_argument('-mt_step_mode', type=str, default='before-after',
                    choices=["exact", "before-after", "before"],
                    help='which rnn states to use when doing mt stuff')
parser.add_argument('-max_entities', type=int, default=2,
                    help='number of distinct entities to predict')
parser.add_argument('-max_mentions', type=int, default=2,
                    help='number of entity tokens to predict')
parser.add_argument('-mt_coeff', type=float, default=1, help='scales mt loss')

parser.add_argument('-emb_size', type=int, default=128, help='size of word embeddings')
parser.add_argument('-rnn_size', type=int, default=128, help='size of rnn hidden state')
parser.add_argument('-feat_size', type=int, default=128, help='')
parser.add_argument('-sp_size', type=int, default=80, help='')
parser.add_argument('-layers', type=int, default=1, help='num rnn layers')
parser.add_argument('-add_inp', action='store_true', help='mlp features (instead of concat)')
parser.add_argument('-dropout', type=float, default=0, help='dropout')
parser.add_argument('-topdrop', action='store_true', help='dropout on last rnn layer')
parser.add_argument('-mt_drop', action='store_true', help='dropout before mt decoder')
parser.add_argument('-relu', action='store_true', help='relu for input mlp')

parser.add_argument('-optim', type=str, default='adam', help='')
parser.add_argument('-lr', type=float, default=0.001, help='learning rate')
parser.add_argument('-epochs', type=int, default=4, help='')
parser.add_argument('-clip', type=float, default=5, help='gradient clipping')
parser.add_argument('-initrange', type=float, default=0.1, help='uniform init interval')
parser.add_argument('-seed', type=int, default=3435, help='')
parser.add_argument('-log_interval', type=int, default=200, help='')

parser.add_argument('-cuda', action='store_true', help='')
args = parser.parse_args()


if __name__ == "__main__":
    print args

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print "WARNING: You have a CUDA device, so you should probably run with -cuda"
        else:
            torch.cuda.manual_seed(args.seed)

    # make data
    data = datastuff.DataStuff(args)

    saved_args, saved_state = None, None
    if len(args.load) > 0:
        saved_stuff = torch.load(args.load)
        saved_args, saved_state = saved_stuff["opt"], saved_stuff["state_dict"]
        net = Reader(data.word_embs, data.words_new2old, saved_args)
        net.load_state_dict(saved_state)
    else:
        args.wordtypes = len(data.words_new2old)
        args.ftypes = data.feat_voc_size
        args.sptypes = data.spee_feat_foc_size
        args.extra_size = data.extra_size
        net = Reader(data.word_embs, data.words_new2old, args)

    data.del_word_embs() # just to save memory

    if args.cuda:
        net = net.cuda()

    optalg = None
    if args.optim == "adagrad":
        optalg = optim.Adagrad(net.parameters(), lr=args.lr)
    else:
        optalg = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999))

    batch_start_idxs = range(0, data.ntrain, args.bsz)
    val_batch_start_idxs = range(0, data.nvalid, args.bsz)

    def train(epoch):
        pred_loss, mt_loss, ndocs = 0, 0, 0
        net.train()
        trainperm = torch.randperm(len(batch_start_idxs))
        for batch_idx in xrange(len(batch_start_idxs)):
            #if batch_idx > 100:
            #    break
            net.zero_grad()
            batch = data.load_data(batch_start_idxs[trainperm[batch_idx]],
                                   args, train=True) # a dict
            bsz = batch["words"].size(1)
            for k in batch:
                batch[k] = Variable(batch[k].cuda() if args.cuda else batch[k])
            word_scores, doc_mt_scores = net(batch)
            lossvar = attn_sum_loss(batch, word_scores)
            pred_loss += lossvar.data[0]
            if args.mt_loss == "idx-loss":
                mt_lossvar = multitask_loss1(batch, doc_mt_scores)
                mt_loss += mt_lossvar.data[0]
                lossvar = lossvar + args.mt_coeff*mt_lossvar
            elif args.mt_loss == "antecedent-loss":
                mt_lossvar = multitask_loss2(batch, doc_mt_scores)
                mt_loss += mt_lossvar.data[0]
                lossvar = lossvar + args.mt_coeff*mt_lossvar
            lossvar /= bsz
            lossvar.backward()
            torch.nn.utils.clip_grad_norm(net.parameters(), args.clip)
            optalg.step()
            ndocs += bsz

            if (batch_idx+1) % args.log_interval == 0:
                print "batch %d/%d | loss %g | mt-los %g" % (batch_idx+1, len(batch_start_idxs),
                                                             pred_loss/ndocs, mt_loss/ndocs)

        print "train epoch %d | loss %g | mt-los %g" % (epoch, pred_loss/ndocs, mt_loss/ndocs)


    def evaluate(epoch):
        total, ncorrect = 0, 0
        for i in xrange(len(val_batch_start_idxs)):
            batch = data.load_data(val_batch_start_idxs[i], args, train=False) # a dict
            bsz = batch["words"].size(1)
            for k in batch:
                batch[k] = Variable(batch[k].cuda() if args.cuda else batch[k], volatile=True)
            word_scores, _ = net(batch)
            ncorrect += get_ncorrect(batch, word_scores)
            total += bsz
        acc = float(ncorrect)/total
        print "val epoch %d | acc: %g (%d / %d)" % (epoch, acc, ncorrect, total)
        return acc

    best_acc = 0
    for epoch in xrange(1, args.epochs+1):
        train(epoch)
        acc = evaluate(epoch)
        if acc > best_acc:
            best_acc = acc
            if len(args.save) > 0:
                print "saving to", args.save
                state = {"opt": args, "state_dict": net.state_dict()}
                torch.save(state, args.save)
        print
