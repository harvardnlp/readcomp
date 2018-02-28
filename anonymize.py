# take input file which contains NER tags and anonymize entities

import numpy as np
import codecs
import re
import sys

sys.stdout = codecs.getwriter('utf8')(sys.stdout)
sys.stderr = codecs.getwriter('utf8')(sys.stderr)

def extract_ner(word):
  match = re.match('(.*)(\/([A-Z]*))(\|\|\|(.*))?', word.strip())
  return match.group(1), match.group(3), match.group(5) if len(match.groups()) >= 5 else None


def replace_speaker(word, speaker_id_list):
  if word in speaker_id_list:
    return 'speaker{}'.format(speaker_id_list.index(word) + 1)
  else:
    return word

with codecs.open('lambadev.txt.lq', 'r', encoding='utf8') as f:
  with codecs.open('lambadev_replace.txt', 'w', encoding='utf8') as wf:
    for line in f:
      tokens = line.split()
      ner_groups = [extract_ner(w) for w in tokens]
      speaker_id_list =[]

      for g in ner_groups:
        if g[1] == 'PERSON':
          if g[0] not in speaker_id_list:
            speaker_id_list.append(g[0])

      line = []
      for g in ner_groups:
        if g[2] is not None:
          line.append(u'{}/{}|||{}'.format(replace_speaker(g[0], speaker_id_list), g[1], replace_speaker(g[2], speaker_id_list)))
        else:
          line.append(u'{}/{}'.format(replace_speaker(g[0], speaker_id_list), g[1]))

      wf.write(u' '.join(line) + '\n')
      
