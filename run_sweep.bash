#!/bin/bash

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
OUTFILE = "$OUTDIR/$OUTDIR-output.console"
if [ -e $OUTFILE ]; then
  echo "Output file " $OUTFILE " already exists, quitting."
  exit
fi
echo "Output File = " $OUTFILE

codefile="train.lua"
echo "Code file = " $codefile

# copy code file and sweep file
mkdir $OUTDIR
cp sweep.bash $OUTDIR/
cp $codefile $OUTDIR/

for i in {1..$NUMGPU} do
	bash $OUTDIR/sweep.bash "$OUTFILE.$i" "$2" "$OUTDIR/$codefile" $i &
done
wait
