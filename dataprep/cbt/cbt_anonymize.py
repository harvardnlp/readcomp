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

def replace_choices(toke, speaker_id_list):
  choices = toke.split('|')
  nuchoices = []
  for choice in choices:
    if choice in speaker_id_list:
      nuchoices.append('speaker{}'.format(speaker_id_list.index(choice)+1))
    else:
      nuchoices.append(choice)
  return '|'.join(nuchoices) 

with codecs.open(sys.argv[1], 'r', encoding='utf8') as f:
  with codecs.open(sys.argv[1]+".replace", 'w', encoding='utf8') as wf:
    for line in f:
      #print line
      tokens = line.split()
      ner_groups = [extract_ner(w) for w in tokens]
      speaker_id_list =[]

      for ii, g in enumerate(ner_groups):
        if ii != len(ner_groups) - 1 and g[0] != 'xxxxx' and g[1] == 'PERSON':
          if g[0] not in speaker_id_list:
            speaker_id_list.append(g[0])

      line = []
      #print ner_groups
      for ii, g in enumerate(ner_groups):
        if ii == len(ner_groups) - 1:
          #line.append(u'{}/O'.format(g[0]))
          line.append(u'{}/O'.format(replace_choices(g[0], speaker_id_list)))
        elif g[2] is not None:
          line.append(u'{}/{}|||{}'.format(replace_speaker(g[0], speaker_id_list), g[1], replace_speaker(g[2], speaker_id_list)))
        else:
          line.append(u'{}/{}'.format(replace_speaker(g[0], speaker_id_list), g[1]))
      #print line
      #assert False
      wf.write(u' '.join(line) + '\n')
