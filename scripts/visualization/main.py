import argparse
import importlib

graphs = ["tld_hist"]
parser = argparse.ArgumentParser()
parser.add_argument("GRAPHS",
                    help = "One or more graphs to generate",
                    nargs = '+',
                    choices = ["all"] + graphs
                )

args = parser.parse_args()
print(args.GRAPHS)

X = None
y = None

if args.GRAPHS[0] == "all":
    modules = graphs
else:
    modules = args.GRAPHS
    
for mod in modules:
    mod = importlib.import_module(mod)
    mod.make_graph(X,y)
