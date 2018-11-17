from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import argparse
import csv

def main():
    parser = argparse.ArgumentParser(description='Visualizes 3D graph indicating time to calculate max common subgraph of 2 graphs')
    parser.add_argument('--approx', '-a', action='store_true', help='was approximation used')
    parser.add_argument('--density', '-d', type=int, help='density used')
    args = parser.parse_args()

    folder = 'results'
    if (args.approx):
        folder = "approx_" + folder
    else:
        folder = "exact_" + folder

    file = folder + "/" + str(args.density) + "_result"

    g1_s = []
    g2_s = []
    t = []
    with open(file + ".csv",'r') as csvfile:
        examples = csv.reader(csvfile, delimiter=',')
        for row in examples:
            g1_s.append(int(row[0]))
            g2_s.append(int(row[1]))
            t.append(float(row[2]) + float(row[3]))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for xs, ys, zs in zip(g1_s, g2_s, t):
        ax.scatter(xs, ys, zs, c='green', marker='.')


    ax.set_xlabel('Size of graph 1')
    ax.set_ylabel('Size of grap 2')
    ax.set_zlabel('Time')

    # plt.show()
    plt.savefig(file + ".png")

if __name__ == '__main__':
    main()