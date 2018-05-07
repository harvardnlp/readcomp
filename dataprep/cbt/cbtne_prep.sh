#!/bin/bash

for split in train valid test; do
    bunzip2 -k NE_$split.txt.ner.bz
    mv NE_$split.txt.ner /tmp/NE_$split.txt.ner
    python cbt_redo_ner.py < /tmp/NE_$split.txt.ner > /tmp/NE_$split.txt.redo
    python to_lamb_fmt.py < /tmp/NE_$split.txt.redo > /tmp/NE_$split.lamb
    python cbt_label_quotes.py < /tmp/NE_$split.lamb > /tmp/NE_$split.lq
    python cbt_anonymize.py /tmp/NE_$split.lq
    mv /tmp/NE_$split.lq.replace .
done
