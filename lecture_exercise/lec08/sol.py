"""
Start for Lecture 8 Exercise
"""

import sys


def get_pixels(fn):
    fp = open(fn)
    if fp is None:
        print("Could not open", fn)
        sys.exit(0)

    triples = []
    for line in fp:
        vals = [int(v) for v in line.strip().split()]
        if len(vals) == 3:
            triples.append(vals)

    return triples


def run(list, t0, t1):
    valid = False
    prints = []
    temp = []
    index = 0

    for item in list:
        index += 1
        if item[2] > t1:
            temp.append(item)
            valid = True
        elif item[2] > t0:
            temp.append(item)
        else:
            if valid:
                prints += temp
                temp.clear()
                valid = False
            else:
                temp.clear()

        if index == len(list) and valid:
            prints += temp

    for i in prints:
        print("%d %d" % (i[0], i[1]))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage:", sys.argv[0], "pts.txt th0, th1")
        sys.exit(0)
    triples = get_pixels(sys.argv[1])
    th0 = int(sys.argv[2])
    th1 = int(sys.argv[3])

    run(triples, th0, th1)

    sys.exit()
