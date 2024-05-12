import sys

args = sys.argv

data_descs = open(args[1]).readlines()
for line in data_descs:
    pair = line.split(sep=":")
    name = pair[0].strip()
    desc = pair[1].strip().strip(".")
    print(f"{name},{desc}")
