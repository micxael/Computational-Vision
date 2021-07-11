import numpy as np
import sys


def build_nn(fn):
    npzfile = np.load(fn)
    a_list = [npzfile[a] for a in npzfile.files]
    mid = len(npzfile.files) // 2
    weights, biases = a_list[:mid], a_list[mid:]
    return weights, biases


def build_xs(fn):
    with open(fn, 'rb') as f:
        xs = np.load(f)
    return xs


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage", sys.argv[0], "root")
        sys.exit(0)

    net_name = sys.argv[1] + "_nn.npz"
    weights, biases = build_nn(net_name)

    x_name = sys.argv[1] + "_x.npy"
    xs = build_xs(x_name)

    layers = [xs.shape[0]] + [len(b) for b in biases]
    # print(xs.shape)
    for w, b in zip(weights, biases):
        # print(w.shape, b.shape)
        xs = np.matmul(w, xs) + b[:, np.newaxis]
        xs[xs < 0] = 0

    print(layers)
    nx = xs.shape[1]
    for i in range(nx):
        x = xs[:, i]
        j_min = np.argmin(x)
        j_max = np.argmax(x)
        print("=====")
        print("Min %d: %.3f" % (j_min, x[j_min]))
        print("Max %d: %.3f" % (j_max, x[j_max]))
