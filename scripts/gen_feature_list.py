import sys

file = open(sys.argv[1])
header = file.readline()
features = header.split(sep = ',')
for feature in features:
    print(f"\"{feature.strip()}\",")
