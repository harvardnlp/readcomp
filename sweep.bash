#!/bin/bash

trap "kill 0" SIGINT # allow to kill all spawning processes in the same group

if [ $1 == "-h" ]; then
  echo "Syntax::sh scriptname output-file num-epochs codefile gpu-id"
  exit
fi

# check outputfile
if [ -z $1 ]; then
  echo "Output file not specified. Using 'output.console' "
  OUTFILE="output.console"
else
  OUTFILE=$1
  if [ -e $OUTFILE ]; then
    echo "Output file " $OUTFILE " already exists, quitting."
    exit
  fi
  echo "Output File = " $OUTFILE
fi

# check number of epochs
if [ -z $2 ]; then
  echo "Number of epochs not specified. Using 3"
  N=3
else
  N=$2
  echo "Num Epochs =" $N
fi

if [ -z $3 ]; then
  echo "code file not specified, using default"
  codefile="train.lua"
else
  codefile=$3
fi

if [ -z $4 ]; then
  echo "gpu device not specified, using 1"
  gpu=1
else
  gpu=$4
fi

if [ -z $5 ]; then
  echo "number of gpus not specified, using 1"
  numgpu=1
else
  numgpu=$5
fi

gpuid=0
seed=(13)
batch=(64)
embed=(64 128 256)
adam=("{0.9, 0.999}")
cutoff=(10)
entity=(2 3 5)
dropout=(0.1 0.2 0)
extra="-datafile lambada.hdf5 -emb_size 128 -std_feats -speaker_feats -maxseqlen 1028 -mt_loss idx-loss -log_interval 1000"
for rs in "${seed[@]}"; do
  for b in "${batch[@]}"; do
    for d0 in "${embed[@]}"; do
      for c in "${cutoff[@]}"; do
        for ent in "${entity[@]}"; do
          for dr in "${dropout[@]}"; do
            gpuid=$((gpuid % numgpu + 1))
            if [ "$gpuid" = "$gpu" ]; then
              printf "iteration = $t: python $codefile -cuda -seed $rs -dropout $dr -bsz $b -epochs $N -rnn_size $d0 -max_entities $ent -max_mentions $ent -clip $c $extra \n" >> $OUTFILE
              python $codefile -cuda -seed $rs -dropout $dr -bsz $b -epochs $N -rnn_size $d0 -max_entities $ent -max_mentions $ent -clip $c $extra >> $OUTFILE
            fi
          done
        done
      done
    done
  done
done

echo "Done"
