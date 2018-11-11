from __future__ import annotations

import io
from functools import partial
from itertools import product

import argparse
from typing import Union, Tuple
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from numpy import genfromtxt


class Graph:
    def __init__(self, matrix: Union[str, io.TextIOWrapper, np.ndarray]):
        if isinstance(matrix, str) or isinstance(matrix, io.TextIOWrapper):
            self.matrix = genfromtxt(matrix, delimiter=',', dtype=int).astype(bool)
        elif isinstance(matrix, np.ndarray):
            self.matrix = np.copy(matrix)
        else:
            raise Exception("Bad argument type")

    def get_neighbors(self, vertex: int) -> np.ndarray:
        """Returns ndarray with indices of neighbors"""
        return np.flatnonzero(self.matrix[vertex])

    def is_connected(self, v1: int, v2: int) -> bool:
        """Check whether vertices are connected"""
        return self.matrix[v1][v2]

    @property
    def n_vertices(self) -> int:
        """Get number of vertices of graph"""
        return self.matrix.shape[0]

    def __str__(self) -> str:
        return str(self.matrix.astype(int))

    @staticmethod
    def modular_product(g1: Graph, g2: Graph) -> Graph:
        """Creates modular product of two graphs(https://en.wikipedia.org/wiki/Modular_product_of_graphs)."""
        n_g1 = g1.n_vertices
        n_g2 = g2.n_vertices
        matrix = np.full([n_g1 * n_g2, n_g1 * n_g2], np.nan)

        encode_index = partial(Graph.encode_index, n=max([n_g1, n_g2]))
        for v1, v2 in product(range(n_g1), range(n_g2)):
            for v1_, v2_ in product(range(n_g1), range(n_g2)):
                if v1 != v1_ or v2 != v2_:
                    matrix[encode_index(v1, v2)][encode_index(v1_, v2_)] = 1 if \
                        (g1.is_connected(v1, v1_) and g2.is_connected(v2, v2_)) or \
                        (not g1.is_connected(v1, v1_) and not g2.is_connected(v2, v2_)) else 0
                else:
                    matrix[encode_index(v1, v2)][encode_index(v1_, v2_)] = 0

        return Graph(matrix.astype(bool))

    @staticmethod
    def decompose_modular_product(modular_g: Graph, vertices_g1: int, vertices_g2: int, verbose=False) -> Tuple[
        Graph, Graph]:
        assert vertices_g1 * vertices_g2 == modular_g.matrix.shape[0]
        encode = partial(Graph.encode_index, n=max([vertices_g1, vertices_g2]))
        matrix1 = np.full([vertices_g1, vertices_g1], np.nan)
        matrix2 = np.full([vertices_g2, vertices_g2], np.nan)

        for i in range(vertices_g1):
            for j in range(vertices_g1):
                if i == j:
                    matrix1[i][j] = 0
                else:
                    matrix1[i][j] = 1 if modular_g.matrix[encode(i, 0)][encode(j, 0)] == 0 else 0
        if verbose: print(matrix1)

        for i in range(vertices_g2):
            for j in range(vertices_g2):
                if i == j:
                    matrix2[i][j] = 0
                else:
                    matrix2[i][j] = 1 if modular_g.matrix[encode(0, i)][encode(0, j)] == 0 else 0
        if verbose: print(matrix2)
        return Graph(matrix1.astype(bool)), Graph(matrix2.astype(bool))

    @staticmethod
    def encode_index(val1: int, val2: int, n: int) -> int:
        """Encodes two values into one. n is max value + 1 of second element in tuple(n_vertices of second)."""
        return val1 * n + val2

    @staticmethod
    def decode_index(val: int, n: int) -> Tuple[int, int]:
        """Returns two values from one. n is max value + 1 of second element in tuple(n_vertices of second)."""
        return val // n, val % n


class MaxClique:
    def __init__(self, G, approx=False):
        self.G = G
        self.approx = approx
        self.max_found = set()
        self.log = []
        self._run()

    def _run(self):
        for i in range(self.G.n_vertices):
            self.log = []
            N = self.G.get_neighbors(i)
            if len(N) >= len(self.max_found):
                U = set()
                for j in N:
                    if j > i:
                        if len(self.G.get_neighbors(j)) >= len(self.max_found):
                            U.add(j)
            print(f"\n\nSearching for i={i}\tU: {U}\tN: {N}")
            if self.approx:
                self._clique_approximation(U, {i})
            else:
                self._clique(U, {i})
            print('Search log:\n{}'.format('\n'.join(self.log)))

        if len(self.max_found) > 0:
            print(f"MAX CLIQUE FOUND:\n{self.max_found}")
        else:
            print("NO CLIQUE FOUND.")

    def _clique(self, U, current_clique):
        self.log.append(f'_clique U: {U}\tcurrent: {current_clique}')
        if len(U) == 0 and len(current_clique) > len(self.max_found):
            self.log.append(f"_new_max_found: {current_clique}\told: {self.max_found}")
            self.max_found = set(current_clique)
            return

        while len(U) > 0:
            if len(current_clique) + len(U) <= len(self.max_found):
                self.log.append(f'_clique_too_small U: {U}\tcurrent: {current_clique}\tmax: {self.max_found}')
                return

            u = U.pop()
            if all([self.G.is_connected(w, u) for w in current_clique]):
                current_clique.add(u)
            N = set(self._filter_neighborhood(u))

            self.log.append(f'_recursive_clique U:{U}\tN:{N}\n\t\t  U&N:{U&N}\tcurrent:{current_clique}\tu:{u}')
            self._clique(U & N, current_clique)

    def _clique_approximation(self, U, current_clique):
        self.log.append(f'_clique U: {U}\tcurrent: {current_clique}')
        if len(U) == 0 and len(current_clique) > len(self.max_found):
            self.log.append(f"_new_max_found: {current_clique}\told: {self.max_found}")
            self.max_found = set(current_clique)
            return

        u = self._get_neighbor_with_max_degree(U)
        if u is not None:
            U.remove(u)
            if all([self.G.is_connected(w, u) for w in current_clique]):
                current_clique.add(u)
            N = set(self._filter_neighborhood(u))

            self.log.append(f'_recursive_clique U:{U}\tN:{N}\n\t\t  U&N:{U&N}\tcurrent:{current_clique}\tu:{u}')
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
        plt.subplot(221)
        visualization = nx.from_numpy_matrix(G.matrix.astype(int))
        pos = nx.spring_layout(visualization)
        nx.draw_networkx(visualization, pos, node_size=100)

        plt.subplot(222)
        nx.draw_networkx(visualization, pos, node_size=50)
        self._outline_selected_vertices(visualization, pos, max_found)

        plt.subplot(223)
        self._draw_original_graph(g1, c1)
        plt.subplot(224)
        self._draw_original_graph(g2, c2)

        plt.show()

    def _outline_selected_vertices(self, visualization, pos, selected_vertex_set):
        nx.draw_networkx_nodes(
            visualization, pos,
            nodelist=list(selected_vertex_set),
            node_color='g',
            node_size=50,
            font_size=6,
        )

    def _draw_original_graph(self, graph, selected_vertex_set):
        visualization = nx.from_numpy_matrix(graph.matrix.astype(int))
        pos = nx.spring_layout(visualization)
        nx.draw_networkx(visualization, pos, node_size=50)
        self._outline_selected_vertices(visualization, pos, selected_vertex_set)


def main() -> None:
    def parse_arguments() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description='Calulates maximal common induced connected subgraph.')
        parser.add_argument('--graphs', '-g', type=argparse.FileType('r'), nargs=2,
                            help='an integer for the accumulator',
                            required=True)

        args = parser.parse_args()
        return args

    args = parse_arguments()
    g1 = Graph(args.graphs[0])
    g2 = Graph(args.graphs[1])
    G = Graph.modular_product(g1, g2)
    print(g1)
    print(g2)

    print("\nLOOKING FOR A MAX CLIQUE IN THE FOLLOWING GRAPH:")
    print(G)
    max_clique = MaxClique(G, approx=False)

    c1, c2 = set(), set()

    for vertex_in_modular_graph in max_clique.max_found:
        v_in_g1, v_in_g2 = Graph.decode_index(vertex_in_modular_graph, g2.n_vertices)
        c1.add(v_in_g1)
        c2.add(v_in_g2)

    Visualizer(g1, g2, G, max_clique.max_found, c1, c2)


if __name__ == '__main__':
    main()
