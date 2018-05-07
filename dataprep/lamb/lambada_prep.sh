#!/bin/bash

for split in train valid test; do
    bunzip2 -k $split.txt.ner.bz
    mv $split.txt.ner /tmp/$split.txt.ner
    python lamb_label_quotes.py < /tmp/$split.txt.ner > /tmp/$split.txt.lq
    python lamb_anonymize.py /tmp/$split.txt.lq
    mv /tmp/$split.txt.lq.replace .
done
