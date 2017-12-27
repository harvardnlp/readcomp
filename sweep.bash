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
class=("asr")
seed=(13)
batch=(64)
embed=(128)
adam=("{0.9, 0.999}")
cutoff=(10)
entity=(2 3 5 10)
dropout=(0.2)
extra="--dontsave --datafile cbt.hdf5 --postsize 128  --nersize 128 --sentsize 80 --speesize 80 --std_feats --ent_feats --disc_feats --speaker_feats --use_choices --lr 0.001 --maxseqlen 1500"
for cls in "${class[@]}"; do
  for rs in "${seed[@]}"; do
    for b in "${batch[@]}"; do
      for d0 in "${embed[@]}"; do
        for ad in "${adam[@]}"; do
          for c in "${cutoff[@]}"; do
            for ent in "${entity[@]}"; do
              for dr in "${dropout[@]}"; do
                gpuid=$((gpuid % numgpu + 1))
                if [ "$gpuid" = "$gpu" ]; then
                  printf "iteration = $t: th $codefile --cuda --device $gpuid --randomseed $rs --model $cls --dropout $dr --batchsize $b --maxepoch $N --hiddensize {$d0} --entity $ent --entitysize $ent --cutoff $c $extra --adamconfig $ad\n" >> $OUTFILE
                  th $codefile --cuda --device $gpuid --randomseed $rs --model $cls --dropout $dr --batchsize $b --maxepoch $N --hiddensize {$d0} --entity $ent --entitysize $ent --cutoff $c $extra --adamconfig "$ad" >> $OUTFILE
                fi
              done
            done
          done
        done
      done
    done
  done
done

echo "Done"
