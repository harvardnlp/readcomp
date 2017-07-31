#!/bin/bash

trap "kill 0" SIGINT # allow to kill all spawning processes in the same group

python preprocess-lambada.py --data ~/data/lambada/lambada-train-valid/small/ --glove ~/data/glove/glove.6B.100d.txt --out_file lambada-small.hdf5
th train.lua --cuda --device 1 --randomseed 101 --model asr --batchsize 64 --postsize 50 --gcepoch 1000 --maxseqlen 1024 --dropout 0.1 --datafile lambada-small.hdf5
th train.lua --testmodel models/lambada-small-asr.t7 --datafile lambada-small.hdf5 --cuda --progress

# python postprocess.py --model_dump_pattern "data\dump\analysis\*analysis*.dump" --analysis_category_file C:\Users\lhoang\Dropbox\Personal\Work\lambada-dataset\lambada_analysis_categories.txt

echo "Done"
