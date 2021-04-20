from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import re
import os
import sys

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

if __name__ == "__main__":
    if len(sys.argv) < 2 or not os.path.isfile(sys.argv[1]):
        print("Usage: \tpython3 gen_plots.py lossfile", file=sys.stderr)
        print("\twhere lossfile is the file containing loss output from colab", file=sys.stderr)
        sys.exit(1)

    filepath = sys.argv[1]
    parent = os.path.dirname(filepath)
    tups = []
    with open(filepath, "r") as f:
        for line in f:
            tups.append(parse_line(line))

    tups = np.array(tups).astype(float)
    if len(sys.argv) > 2:
        plot_data(tups, parent, sys.argv[2])
    else:
        plot_data(tups, parent)


