import sys

# ALSO LOWERCASES, BRO

prons = set(["he", "they", "she", "we", "i"])


def get_wrd_label(toke):
    """
    necessary b/c some words have /'s in them
    """
    pieces = toke.split('/')
    if len(pieces) == 2:
        return pieces
    else:
        labe = pieces[-1]
        wrd = "".join(pieces[:-1]) # should really be '/'.join() but doesn't matter
        return wrd, labe


def stupid_label(tokens):
    """
    assumes tokens have NER tags
    """
    labels = []
    last_start, last_name = 0, None
    prev_was_close = False # sometimes closes are used to start???
    try:
        first_open = tokens.index("``/O")
    except ValueError:
        first_open = len(tokens)
    try:
        first_close = tokens.index("''/O")
    except ValueError:
        first_close = len(tokens)
    in_quote = first_close < first_open
    for i, toke in enumerate(tokens):
        if toke == "''/O" and not prev_was_close: # end quote
            if tokens[i-1] != "./O":
                # find closest name (w/in ten tokens, let's say)
                name = None
                for jj in xrange(i+1, min(i+1+10, len(tokens))):
                    wrd, labe = get_wrd_label(tokens[jj])
                    wrd = wrd
                    if labe == "PERSON" or wrd in prons:
                        name = wrd
                        break
            else:
                name = last_name
            if name is not None:
                labels.append((last_start, i, name))
                last_name = name
            in_quote = False
            prev_was_close = True
        elif toke == "``/O" or toke == "''/O" and prev_was_close: # start quote
            in_quote = True
            last_start = i
            prev_was_close = False # hmmm
        elif not in_quote and ("PERSON" in toke or toke.split('/')[0] in prons):
            last_name = toke.split('/')[0]
    if in_quote and last_name is not None:
        labels.append((last_start, len(tokens)-1, last_name))

    tokencopy = [toke for toke in tokens]
    for (start, end, name) in labels:
        for k in xrange(start+1, end):
            if "|||" in tokens[k]:
                print
                print
                print " ".join(tokencopy)
                print labels
                print k, tokens[k]
            assert "|||" not in tokens[k]
            assert "|||" not in name
            tokens[k] = tokens[k] + "|||" + name

    return tokens


for line in sys.stdin:
    tokes = line.strip().split()
    lowertokes = []
    for ii in xrange(len(tokes)):
        wrd, labe = get_wrd_label(tokes[ii])
        lowertokes.append("%s/%s" % (wrd.lower(), labe))
    assert len(lowertokes) == len(tokes)
    labeled = stupid_label(lowertokes)
    print " ".join(labeled)
