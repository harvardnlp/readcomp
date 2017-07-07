#!/bin/bash

if [ $1 == "-h" ]; then
  echo "Syntax::sh scriptname output-file num-epochs"
  exit
fi

# check outputfile
if [ -z $1 ]; then
  echo "Output file not specified. Using 'output.console' "
  OUTFILE="output.console"
else
  OUTFILE=$1
fi

OUTDIR="sweep-$OUTFILE"
OUTFILE = "$OUTDIR/$OUTFILE"
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

bash $OUTDIR/sweep.bash $OUTFILE "$2" "$OUTDIR/$codefile"