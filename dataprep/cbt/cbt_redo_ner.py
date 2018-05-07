import sys

animate_grams_path = "animate.unigrams.txt"

animates = set()
with open(animate_grams_path) as f:
    for line in f:
        animates.add(line.strip())
        #animates.add(line.strip().lower())

titles = set(["farmer", "king", "queen", "aunt", "uncle", "cousin", "mother", 
              "father", "prince", "princess", "st.", "lady", "lord", "mr", "mrs",
              "mr.", "mrs.", "sir", "madame", "madam", "young", "old", "little", "happy",
              "mistah", "big", "fat", "wild", "long", "de", "brother", "sister"])

def o_tag(toke):
   slidx = toke.rfind('/')
   nutoke = "%s/O" % toke[:slidx]
   return nutoke
 
def get_word(toke):
    slidx = toke.rfind('/')
    return toke[:slidx]

for line in sys.stdin:
    tokes = line.strip().split()
    tokes[-1] = o_tag(tokes[-1]) # last thing is always the choices
    for i, toke in enumerate(tokes):
        if get_word(toke).lower() == "the" and get_word(tokes[i+1]) in animates and (i+2 >= len(tokes) or "PERSON" not in tokes[i+2]): #get_word(tokes[i+1]).lower() in animates:
            slidx = tokes[i+1].rfind('/')
            nutoke = "%s/PERSON" % tokes[i+1][:slidx]
            tokes[i+1] = nutoke
        elif "/PERSON" in toke:
            j = i+1
            while j < len(tokes) and "/PERSON" in tokes[j]:
                j += 1
            if j-i > 1: # multiword person
                first = tokes[i].split('/')[0]
                if first.lower() in titles: # use second word as the real name
                    tokes[i] = o_tag(tokes[i])
                    for k in xrange(i+2, j):
                        tokes[k] = o_tag(tokes[k])
                else: # use first word as real name
                    for k in xrange(i+1, j):
                        tokes[k] = o_tag(tokes[k])
    print " ".join(tokes)
