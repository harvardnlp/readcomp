#!/bin/bash

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

gpuid=0
class=("asr")
seed=(7)
batch=(64)
embed=(128 58 200)
adam=("{0, 0.999}")
cutoff=(10)
post=(80 65 70 75)
extra=""
for cls in "${class[@]}"; do
  for rs in "${seed[@]}"; do
    for b in "${batch[@]}"; do
      for d0 in "${embed[@]}"; do
        for ad in "${adam[@]}"; do
          for c in "${cutoff[@]}"; do
            for pst in "${post[@]}"; do
              gpuid=$((gpuid % 2 + 1))
              if [ $gpuid -eq $gpu ] then
                printf "iteration = $t: th $codefile --cuda --device $gpuid --progress --randomseed $rs --model $cls --batchsize $b --maxepoch $N --adamconfig $ad --hiddensize {$d0} --postsize $pst --cutoff $c $extra\n" >> $OUTFILE
                th $codefile --cuda --device $gpuid --progress --randomseed $rs --model $cls --batchsize $b --maxepoch $N --adamconfig $ad --hiddensize {$d0} --postsize $pst --cutoff $c $extra >> $OUTFILE
              fi
            done
          done
        done
      done
    done
  done
done

echo "Done"
