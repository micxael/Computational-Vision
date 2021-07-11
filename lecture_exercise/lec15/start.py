import numpy as np
import sys
from itertools import product

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

    """
    weight_matrix = np.vstack(weights).T
    biases_matrix = np.vstack(biases).T
    xs_matrix = np.vstack(xs).T
    """
    xs = np.vstack(xs).T.tolist()
    print("xs\n")
    print(xs)
    print("\n")
    print("weight\n")
    print(weights)
    print("\n")
    print("biases\n")
    print(biases)
    print("----------------------")
    print("\n\n\n")

    temp = []
    for input_vector in xs:
        temp.clear()
        temp = input_vector
        i = 0
        j = 1
        for weight in weights:
            temp2 = temp.copy()
            temp.clear()
            for weight2 in weight:
                print(j, end=' ')
                j +=1
                print("xs")
                print(temp2)
                print("weight")
                print(weight2)
                temp.append(sum([x * y for x, y in zip(temp2, weight2)]))
                print("temp:")
                print(temp)
                print("\n")
            temp = [x + y for x, y in zip(temp, biases[i])]
            temp = [0 if i < 0 else i for i in temp]
            print("----------biases-----------")
            print(biases[i])
            print("-------final temp----------")
            print(temp)
            print("---------------------------\n")
            i+=1






    """
    temp = []
    temp.append(len(weights[0][0]))
    for bias in biases:
        temp.append(len(bias))
    print(temp)
    """
    sys.exit()
