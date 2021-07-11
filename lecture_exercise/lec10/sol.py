import math
import sys


def angle_interpolation(Ix, Iy):
    deg = math.degrees(math.atan2(Iy, Ix))

    if deg < 0:
        deg += 360

    index = deg / 45
    index1 = math.floor(index)

    if index1 + 0.5 < index:
        index2 = math.ceil(index) % 8
    else:
        index2 = (index1 - 1) % 8


    val = index - (index1 + 0.5)
    val = round(val, 2)
    if val < 0:
        val *= -1

    print("%d %.2f; %d %.2f" % (int(index1), 1 - val, int(index2), val))


def location_interpolation(x):
    val = x - (math.floor(x) + 0.5)
    if val > 0:
        index1 = math.floor(x)
        index2 = index1 + 1
    else:
        index1 = math.floor(x)
        index2 = index1 -1

    if val < 0:
        val *= -1

    if 0 <= index1 < 4 and 0 <= index2 < 4:
        print("%1d %.2f; %1d %.2f" % (int(index1), 1 - val, int(index2), val))

    elif 0 <= index1 < 4:
        print("%1d %.2f" % (int(index1), 1 - val))
    else:
        print("%1d %.2f" % (int(index2), val))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage:", sys.argv[0], "pts.txt")
        sys.exit(0)
    pf = open(sys.argv[1])

    n = 0
    for line in pf:
        line = line.strip().split()
        if len(line) != 4:
            continue
        x, y, Ix, Iy = [float(v) for v in line]

        n += 1
        print("======\nPoint", n)
        angle_interpolation(Ix, Iy)
        location_interpolation(x)
        location_interpolation(y)

    sys.exit()
