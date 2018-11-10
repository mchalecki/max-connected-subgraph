from __future__ import annotations

import io
from functools import partial
from itertools import product

import argparse
from typing import Union, Tuple
import numpy as np

from numpy import genfromtxt


class Graph:
    def __init__(self, matrix: Union[str, io.TextIOWrapper, np.ndarray]) -> None:
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
    def n_vertices(self)->int:
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

        get_index = partial(Graph.get_index, n=n_g2)
        for v1, v2 in product(range(n_g1), range(n_g2)):
            for v1_, v2_ in product(range(n_g1), range(n_g2)):
                matrix[get_index(v1, v2)][get_index(v1_, v2_)] = 1 if \
                    (g1.is_connected(v1, v1_) and g2.is_connected(v2, v2_)) or \
                    (not g1.is_connected(v1, v1_) and not g2.is_connected(v2, v2_)) else 0
        return Graph(matrix)

    @staticmethod
    def encode_index(val1: int, val2: int, n: int) -> int:
        """Encodes two values into one. n is max value + 1 of second element in tuple(n_vertices of second)."""
        return val1 * n + val2

    @staticmethod
    def decode_index(val: int, n: int) -> Tuple[int, int]:
        """Returns two values from one. n is max value + 1 of second element in tuple(n_vertices of second)."""
        return val // n, val % n


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
    g = Graph.modular_product(g1, g2)
    print(g)


if __name__ == '__main__':
    main()
