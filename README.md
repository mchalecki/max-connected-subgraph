# Maximal Common Indeuced Subgraph
### Authors:
- Marcin Chalecki
- Adam Przywarty
- Maciej Korzeniewski

## Definition of the problem
Given two graphs G1 and G2 defined as: 

G1=(V1, E1) , G2=(V2, E2)   

find a maximal common connected subgraph.
Assume that

|V1| > |V2|.

A common subgraph is a graph G=(V,E) such that it is a subgraph of G1 and G2.

We have to take n vertices from each graph and take all edges connecting those n vertices (if vertex has an edge in initial graph, then it must have an edge in our result). Finally, we have to find such subgraphs which are isomorphic (they have the same number of vertices, edges and have the same edge connectivity).
