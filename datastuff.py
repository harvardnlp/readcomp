import torch

class DataStuff(object):

    def __init__(self):
        self.word_ctx = torch.LongTensor()
        self.answers = torch.LongTensor()
        self.linenos = torch.LongTensor()
        self.feats = torch.LongTensor()
        self.extr = torch.Tensor()
        self.spee_feats = torch.LongTensor()
        self.choices = torch.Tensor()

    def load_data(self, dat, batch_idx, args, train=True):
        """
        dat is a dict w/ all the data stuff
        batch_idx is the idx of first thing in the batch
        """
        pfx = "train" if train else "valid"
        loc = dat["%s_location" % pfx] # nexamples x 3
        bsz = min(args.bsz, loc.size(0)-batch_idx)
        max_ctx_len = min(args.maxseqlen, loc[batch_idx:batch_idx+bsz, 1].max())
        self.word_ctx.resize_(max_ctx_len, bsz).zero_()
        self.answers.resize_(bsz).zero_()
        self.linenos.resize_(bsz).zero_()
        assert False # check whether 0 is really pad...(since we zero above)

        if args.std_feats:
            self.feats.resize_(max_ctx_len, bsz, 3).zero_()
            self.extr.resize_(max_ctx_len, bsz, args.extra_size).zero_()
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
                assert False # need to add to feat idxs so we don't overlap
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
                    dat["%s_sid" % pfx][answer_idx-capped_len:answer_idx])
                self.spee_feats[-capped_len:, b, 1].copy_(
                    dat["%s_speech" % pfx][answer_idx-capped_len:answer_idx])

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
        return batch
