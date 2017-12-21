import sys
import subprocess
import random

import argparse

opts = [
    #("add_inp", [True]),
    ("clip", [1.0, 5.0, 10.0]),
    ("dropout", [0, 0.1, 0.2, 0.3, 0.4, 0.5]),
    ("emb_size", [100, 128, 200, 256, 300]),
    ("lr", [0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]),
    ("beta1", [0.5, 0.7, 0.9]),
    ("max_entities", [2, 3, 5, 10]),
    ("max_mentions", [2, 3, 5, 20, 100]),
    #("-mt_loss", ["ant-loss", "idx-loss"]),
    ("mt_coeff", [0.5, 1, 1.5]),
    ("mt_drop", [True, False]),
    #("mt_step_mode", ["before"]),
    ("relu", [True, False]),
    ("use_choices", [True, False]),
    ("use_qidx", [True, False]),
    #("-transform_for_ants", [True]),
    #("epochs", [3])
    ]

parser = argparse.ArgumentParser(description='')
parser.add_argument('-ntrials', type=int, default=1000, help='')
parser.add_argument('-seed', type=int, default=0, help='')
parser.add_argument('-gpuid', type=int, default=0, help='')
parser.add_argument('-mt_loss', type=str, default='',
                    choices=["", "idx-loss", "ant-loss"], help='')
parser.add_argument('-mt_step_mode', type=str, default='before',
                    choices=["exact", "before-after", "before"],
                    help='which rnn states to use when doing mt stuff')
parser.add_argument('-logfile', type=str, default='')

args = parser.parse_args()

random.seed(args.seed)

cmd = "CUDA_VISIBLE_DEVICES=%(gpuid)d python train2.py -datafile NE_cbt.hdf5 -dropout %(dropout)g "\
      "-std_feats -speaker_feats -emb_size %(emb_size)d -rnn_size %(emb_size)d -log_interval 1000 "\
      "-bsz 64 -add_inp -cuda -clip %(clip)g %(use_choices)s -maxseqlen 1500 %(use_qidx)s "\
      "%(mt_loss)s -max_entities %(max_entities)d -max_mentions %(max_mentions)d "\
      "-transform_for_ants -mt_step_mode %(mt_step_mode)s -beta1 %(beta1)g -mt_coeff %(mt_coeff)g "\
      "%(relu)s %(mt_drop)s -lr %(lr)g -epochs 4 | tee -a %(logfile)s"

for i in xrange(args.ntrials):
    # randomly sample stuff
    params = {}
    for key, optlist in opts:
        # sample a val
        val = random.choice(optlist)
        if isinstance(val, bool) and val:
            val = "-" + key
        if key == "use_choices" and not isinstance(val, str): # was False
            val = "-use_test_choices"
        if not isinstance(val, bool):
            params[key] = val
        else:
            params[key] = ""

    # fill in the rest of the stuff
    params["gpuid"] = args.gpuid
    if args.mt_loss == "":
        params["mt_loss"] = ""
    else:
        params["mt_loss"] = "-mt_loss %s " % args.mt_loss
    params["mt_step_mode"] = args.mt_step_mode
    params["logfile"] = args.logfile
    pcmd = cmd % params
    print "running", pcmd
    try:
        subprocess.call(pcmd, shell=True)
    except Exception as ex:
        print "couldn't run", ex
    print
    print
