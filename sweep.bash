#!/bin/bash

if [ $1 == "-h" ]; then
  echo "Syntax::sh scriptname output-file num-epochs codefile"
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

gpuid=1
class=("asr")
seed=(7)
batch=(64)
embed=(128)
adam=("{0, 0.999}")
cutoff=(10 5)
post=(80)
extra=""
for cls in "${class[@]}"; do
  for rs in "${seed[@]}"; do
    for b in "${batch[@]}"; do
      for d0 in "${embed[@]}"; do
        for ad in "${adam[@]}"; do
          for c in "${cutoff[@]}"; do
            for pst in "${post[@]}"; do
              printf "iteration = $t: th $codefile -classifier $cls -N $N -startlr $eta -D0 $d0 -save -saveminacc 0.35 $extra\n" >> $OUTFILE
              th $codefile --cuda --device $gpuid --progress --randomseed $rs --model $cls --batchsize $b --maxepoch $N --adamconfig $ad --hiddensize {$d0} --postsize $pst --cutoff $c $extra >> $OUTFILE
            done
          done
        done
      done
    done
  done
done

echo "Done"
