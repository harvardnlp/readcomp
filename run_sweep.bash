#!/bin/bash

trap "kill 0" SIGINT # allow to kill all spawning processes in the same group

if [ $1 == "-h" ]; then
  echo "Syntax::sh scriptname sweep-name num-epochs num-gpus"
  exit
fi

# check outputfile
if [ -z $1 ]; then
  echo "Sweep name not specified. Using 'default' "
  SWEEP="default"
else
  SWEEP=$1
fi

if [ -z $3 ]; then
  echo "Number of gpus not specified. Using 1 "
  NUMGPU=1
else
  NUMGPU=$3
fi

OUTDIR="sweep-$SWEEP"
OUTFILE="$OUTDIR/output-$SWEEP.console"
echo "Using output dir " $OUTDIR
echo "Using output file " $OUTFILE

if [ -e $OUTFILE ]; then
  echo "Output file " $OUTFILE " already exists, quitting."
  exit
fi
echo "Output File = " $OUTFILE

codefile="nutrain.lua"
echo "Code file = " $codefile

# copy code file and sweep file
mkdir $OUTDIR
cp sweep.bash $OUTDIR/
cp -rf $codefile $OUTDIR/
cp -rf *.bash $OUTDIR/
cp -rf *.lua $OUTDIR/
cp -rf *.py $OUTDIR/

for (( i=1; i<=$NUMGPU; i++ )); do
	bash $OUTDIR/sweep.bash "$OUTFILE.$i" "$2" "$OUTDIR/$codefile" $i $NUMGPU &
done
wait
