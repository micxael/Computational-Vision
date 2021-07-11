"""
Starting code for the Lecture 13 exercise.
"""

import sys
import numpy as np


def read_votes(f):
    votes = []
    for line in f:
        line = line.strip().split()
        line = [int(s) for s in line]
        votes.append(line)
    return np.array(votes)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage:", sys.argv[0], "votes.txt")
        sys.exit(0)

    f = open(sys.argv[1], 'r')
    if f is None:
        print("Failed to open", sys.argv[1])
        sys.exit(0)

    votes = read_votes(f)

    scheme_1 = votes.sum(axis=0)
    print("Class %d: %.2f" % (np.argmax(scheme_1), round(max(scheme_1) / sum(scheme_1), 2)))


    scheme_2 = votes / votes.sum(axis=1)[:, None]
    # print(scheme_2)
    print("Class %d: %.2f" % (np.argmax(sum(scheme_2)[:]), round(max(sum(scheme_2)[:]) / scheme_2.shape[0], 2)))


    """
    scheme_3_votes = votes.argmax(axis=1)
    votecounts = np.bincount(scheme_3_votes)
    
    this could work but doesn't account for ties
    """
    votecounts = np.zeros(votes.shape[1])
    for x in range(votes.shape[0]):
        max_val = max(votes[x, :])
        index_list = []
        for i, j in enumerate(votes[x, :]):
            if j == max_val:
                index_list.append(i)
        for idx in index_list:
            votecounts[idx] += 1 / len(index_list)

    print("Class %d: %.2f" % (np.argmax(votecounts), round(max(votecounts) / votes.shape[0], 2)))
    sys.exit()
