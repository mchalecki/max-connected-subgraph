import csv
import io
import time
import os
from functools import partial
from itertools import product

import argparse
from tqdm import tqdm
from typing import Union, Tuple, List
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from numpy import genfromtxt


class Graph:
    def __init__(self, matrix):
        if isinstance(matrix, str) or isinstance(matrix, io.TextIOWrapper):
            self.matrix = genfromtxt(matrix, delimiter=',', dtype=int).astype(bool)
        elif isinstance(matrix, np.ndarray):
            self.matrix = np.copy(matrix)
        else:
            raise Exception("Bad argument type")

    def get_neighbors(self, vertex):
        """Returns ndarray with indices of neighbors"""
        return np.flatnonzero(self.matrix[vertex])

    def is_connected(self, v1, v2):
        """Check whether vertices are connected"""
        return self.matrix[v1][v2]

    @property
    def n_vertices(self):
        """Get number of vertices of graph"""
        return self.matrix.shape[0]

    def __str__(self):
        return str(self.matrix.astype(int))

    @staticmethod
    def modular_product(g1, g2):
        """Creates modular product of two graphs(https://en.wikipedia.org/wiki/Modular_product_of_graphs)."""
        n_g1 = g1.n_vertices
        n_g2 = g2.n_vertices
        matrix = np.full([n_g1 * n_g2, n_g1 * n_g2], np.nan)

        encode_index = partial(Graph.encode_index, n_vertices=[n_g1, n_g2])
        for v1, v2 in tqdm(product(range(n_g1), range(n_g2))):
            for v1_, v2_ in product(range(n_g1), range(n_g2)):
                if v1 != v1_ and v2 != v2_:
                    matrix[encode_index(v1, v2)][encode_index(v1_, v2_)] = 1 if \
                        (g1.is_connected(v1, v1_) and g2.is_connected(v2, v2_)) or \
                        (not g1.is_connected(v1, v1_) and not g2.is_connected(v2, v2_)) else 0
                else:
                    matrix[encode_index(v1, v2)][encode_index(v1_, v2_)] = 0

        return Graph(matrix.astype(bool))

    @staticmethod
    def encode_index(val1, val2, n_vertices):
        """Encodes two values into one. n_vertices is list of number of vertices in graphs."""
        n = n_vertices[1]
        return val1 * n + val2

    @staticmethod
    def decode_index(val, n_vertices):
        """Returns two values from one. n_vertices is list of number of vertices in graphs."""
        n = n_vertices[1]
        return val // n, val % n


class MaxClique:
    def __init__(self, G, approx=False, verbose=False):
        self.G = G
        self.approx = approx
        self.verbose = verbose
        self.max_found = set()
        self.log = []
        self._run()

    def _run(self):
        for i in tqdm(range(self.G.n_vertices)):
            self.log = []
            N = self.G.get_neighbors(i)
            if len(N) >= len(self.max_found):
                U = set()
                for j in N:
                    if j > i:
                        if len(self.G.get_neighbors(j)) >= len(self.max_found):
                            U.add(j)
            self.verbose and print(f"\n\nSearching for i={i}\tU: {U}\tN: {N}")
            if self.approx:
                self._clique_approximation(U, {i})
            else:
                self._clique(U, {i})
            self.verbose and print('Search log:\n{}'.format('\n'.join(self.log)))

        if len(self.max_found) > 0:
            print(f"MAX CLIQUE FOUND:\n{self.max_found}")
        else:
            print("NO CLIQUE FOUND.")

    def _clique(self, U, current_clique):
        self.verbose and self.log.append(f'_clique U: {U}\tcurrent: {current_clique}')
        if len(U) == 0 and len(current_clique) > len(self.max_found):
            self.verbose and self.log.append(f"_new_max_found: {current_clique}\told: {self.max_found}")
            self.max_found = set(current_clique)
            return

        while len(U) > 0:
            if len(current_clique) + len(U) <= len(self.max_found):
                self.verbose and self.log.append(f'_clique_too_small U: {U}\tcurrent: {current_clique}\tmax: {self.max_found}')
                return

            u = U.pop()
            if all([self.G.is_connected(w, u) for w in current_clique]):
                current_clique.add(u)
            N = set(self._filter_neighborhood(u))

            self.verbose and self.log.append(f'_recursive_clique U:{U}\tN:{N}\n\t\t  U&N:{U&N}\tcurrent:{current_clique}\tu:{u}')
            self._clique(U & N, current_clique)

    def _clique_approximation(self, U, current_clique):
        self.verbose and self.log.append(f'_clique U: {U}\tcurrent: {current_clique}')
        if len(U) == 0 and len(current_clique) > len(self.max_found):
            self.verbose and self.log.append(f"_new_max_found: {current_clique}\told: {self.max_found}")
            self.max_found = set(current_clique)
            return

        u = self._get_neighbor_with_max_degree(U)
        if u is not None:
            U.remove(u)
            if all([self.G.is_connected(w, u) for w in current_clique]):
                current_clique.add(u)
            N = set(self._filter_neighborhood(u))

            self.verbose and self.log.append(f'_recursive_clique U:{U}\tN:{N}\n\t\t  U&N:{U&N}\tcurrent:{current_clique}\tu:{u}')
            self._clique_approximation(U & N, current_clique)

    def _get_neighbor_with_max_degree(self, U):
        max_degree = 0
        max_neighbor = None
        for u in U:
            degree = len(self.G.get_neighbors(u))
            if degree > max_degree:
                max_neighbor = u
                max_degree = degree
        return max_neighbor

    def _filter_neighborhood(self, u):
        return [w for w in self.G.get_neighbors(u) if len(self.G.get_neighbors(w)) >= len(self.max_found)]


class Visualizer:
    def __init__(self, g1, g2, G, max_found, c1, c2):
        ax = plt.subplot(131)
        ax.set_title("First graph (green is the maximal common subgraph)", fontsize=8)
        visualization = nx.from_numpy_matrix(g1.matrix.astype(int))
        pos = nx.spring_layout(visualization)
        nx.draw_networkx(visualization, pos, node_size=100)
        self._outline_selected_vertices(visualization, pos, c1)
        plt.axis('off')

        ax = plt.subplot(132)
        ax.set_title("Second graph (green is the maximal common subgraph)", fontsize=8)
        visualization = nx.from_numpy_matrix(g2.matrix.astype(int))
        pos = nx.spring_layout(visualization)
        nx.draw_networkx(visualization, pos, node_size=100)
        self._outline_selected_vertices(visualization, pos, c2)
        plt.axis('off')

        ax = plt.subplot(133)
        ax.set_title("Modular product of both graphs. Green is the maximal clique found.", fontsize=8)
        visualization = nx.from_numpy_matrix(G.matrix.astype(int))
        pos = nx.spring_layout(visualization)
        nx.draw_networkx(visualization, pos, node_size=50)
        self._outline_selected_vertices(visualization, pos, max_found)
        plt.axis('off')

        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
        plt.show()

    def _outline_selected_vertices(self, visualization, pos, selected_vertex_set):
        nx.draw_networkx_nodes(
            visualization, pos,
            nodelist=list(selected_vertex_set),
            node_color='g',
            node_size=100,
            font_size=6,
        )

    def _draw_original_graph(self, graph, selected_vertex_set):
        visualization = nx.from_numpy_matrix(graph.matrix.astype(int))
        pos = nx.spring_layout(visualization)
        nx.draw_networkx(visualization, pos, node_size=50)
        self._outline_selected_vertices(visualization, pos, selected_vertex_set)


def main():
    parser = argparse.ArgumentParser(description='Calulates maximal common induced connected subgraph.')
    parser.add_argument('--verbose', '-v', action='store_true', help='print log messages (for debugging).')
    parser.add_argument('--approx', '-a', action='store_true', help='use faster approximating algorithm.')
    parser.add_argument('--input', '-i', type=str, help='input directory.')
    parser.add_argument('--num', '-n', type=int, choices=range(1, 13), help='choose which test to run.')
    args = parser.parse_args()
    print(args.num)

    data = os.listdir(args.input)
    data.sort(key=int)

    if args.num:
        example = data[args.num-1]
        calculate_example(example, args)
        return

    # if args.approx:
    #     for example in data[6:]:
    #         calculate_example(example, args)
    # else:
    #     for example in data[:6]:
    #         calculate_example(example, args)
    for example in data:
        calculate_example(example, args)


def calculate_example(example, args):
    print(f'\t\t+++STARTING {example}+++\n')
    graphs = os.listdir(f'{args.input}/{example}')
    get_path = lambda name: os.path.realpath('.') + f'/{args.input}/{example}/{name}'
    g1 = Graph(get_path(graphs[0]))
    g2 = Graph(get_path(graphs[1]))
    print(f'g1:\n {g1} \n\n g2:\n{g2}')
    start = time.time()
    G = Graph.modular_product(g1, g2)
    duration1 = time.time() - start
    print(f'\n\t+++MODULAR PRODUCT FOUND IN: {duration1}.s+++\n')

    args.approx and print('\t\t+++APPROXIMATING+++')
    print("\n\t+++LOOKING FOR A MAX CLIQUE IN THE FOLLOWING GRAPH:+++")
    print(G)

    start = time.time()
    max_clique = MaxClique(G, approx=args.approx, verbose=args.verbose)
    duration2 = time.time() - start
    print(f'\n\t+++MAX CLIQUE FOUND IN: {duration2}.s+++\n')

    c1 = []
    c2 = []

    for v in max_clique.max_found:
        i, j = Graph.decode_index(v, [g1.n_vertices, g2.n_vertices])
        c1.append(i)
        c2.append(j)

    print(f'\n\t+++SUBGRAPH IN G1+++')
    print('\t\t'+str(c1))
    print(f'\n\t+++SUBGRAPH IN G2+++')
    print('\t\t'+str(c2))

    print(f'\n\t+++PLEASE CLOSE THE WINDOW TO CONTINUE...+++')
    # Visualizer(g1, g2, G, max_clique.max_found, c1, c2)
    fields=[g1.n_vertices, g2.n_vertices, duration1, duration2]
    with open(str(args.input) + '_result.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fields)


if __name__ == '__main__':
    main()
