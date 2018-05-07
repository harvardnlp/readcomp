import sys

buf = []

for line in sys.stdin:
    buf.append(line.strip())
    if len(buf) >= 22:
        assert len(buf) == 22
        for i, sent in enumerate(buf):
            if i < 21: # get rid of stupid number
                print " ".join(sent.split()[1:]),
            else:
                print " ".join(sent.split())
        buf = []
assert len(buf) == 0
