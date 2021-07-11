import sys
import numpy as np


def read_results(f):
    C = []
    B = []
    for line in f:
        line = line.strip().split()
        if len(line) == 0:
            continue
        C.append(int(line[0]))
        line = [int(s) for s in line[1:]]
        B.append(line)
    return np.array(C), np.array(B)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage:", sys.argv[0], "result.txt")
        sys.exit(0)

    f = open(sys.argv[1], 'r')
    if f is None:
        print("Failed to open", sys.argv[1])
        sys.exit(0)

    C, B = read_results(f)


    holder = np.zeros(B.shape)
    holder = np.cumsum(B, axis=1)
    divider = np.arange(1, B.shape[1] + 1)

    holder = holder / divider * B

    C = np.sum(holder, axis=1) / C

    print(round(C[0], 3))
    print(round(np.sum(C) / len(C), 3))
