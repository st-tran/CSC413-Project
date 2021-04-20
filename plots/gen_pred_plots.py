from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import re
import os
import sys
from glob import glob

def parse_line(line: str) -> Tuple[float]:
    """ Returns 3-tuple (train loss, val loss, val accuracy) """
    return tuple(re.findall(r"([0-9]+\.[0-9]+)", line))

def plot_data(tups: np.ndarray, dir: str=None, title: str=None) -> None:
    iters = np.arange(tups.shape[0])
    plotname = os.path.basename(dir)

    if title is None:
        title = plotname

    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Negative log likelihood loss")
    ax1.plot(iters, tups[:,0], label="Train loss")
    ax1.plot(iters, tups[:,1], label="Validation loss")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy")
    ax2.plot(iters, tups[:,2], label="Validation accuracy", color="g")

    ax1.legend(loc=2)
    ax2.legend(loc=1)
    plt.title(f"{title} loss and accuracy curves")

    if dir and os.path.isdir(dir):
        fig.savefig(f"{dir}/{plotname}-curves")
    plt.show()

def parse_file(f):
    what = {}
    for line in f.readlines():
        line = line.strip()
        province = line.split()[0]
        accuracy = line.split()[-1]
        what[province] = float(accuracy)
    return what

def subcategorybar(X, vals, models, width=0.8):
    n = len(vals)
    _X = np.arange(len(X))
    for i in range(n):
        plt.bar(_X - width/2. + i/float(n)*width, vals[i], 
                width=width/float(n), align="edge", label=models[i])
    plt.xticks(_X, X, rotation=90)
    plt.legend()

if __name__ == "__main__":
    data = {}
    for dir in [d[0] for d in os.walk(".") if ".idea" not in d[0] and d[0] != "." and d[0].startswith("./efficientnet")]:
        with open(f"{dir}/preds", "r") as f:
            data[os.path.basename(dir)] = parse_file(f)
    models = list(data.keys())
    initial = len(data[models[0]].keys())
    for model in models:
        if initial != len(data[model].keys()):
            print("Data is bad", file=sys.stderr)
            sys.exit(1)

    provinces = sorted(list(k for k in data[model].keys() if k != "Overall"))
    accs = [[data[model][province] for province in provinces] for model in models]
    subcategorybar(provinces, accs, models)
    plt.title("EfficientNet predictions on test set")
    plt.tight_layout()
    plt.savefig("efficientnet")