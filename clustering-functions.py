'''
Many real world grpahs are not fully connected. They can be described as small world networks. In such cases, we can use the concept of clustering to 
find the connected components in the graph.

The clustering coefficient of a node is the ratio of the number of edges between the neighbors of the node 
to the total number of possible edges between the neighbors of the node:

    C = 2 * E / (k * (k - 1))

Where E is the number of edges between the neighbors of the node and k is the number of neighbors of the node.

The clustering coefficient of a graph is the average of the clustering coefficients of all the nodes in the 
graph:

    C = 1 / n * sum(C_i)

'''
import networkx as nx
import matplotlib.pyplot as plt

def clustering_coefficient(G, node):
    neighbors = list(G.neighbors(node))
    k = len(neighbors)
    if k < 2:
        return 0
    
    possible_edges = k * (k - 1) / 2
    actual_edges = 0

    for i, u in enumerate(neighbors):
        for j, v in enumerate(neighbors):
            if i < j and G.has_edge(u, v):
                actual_edges += 1

    coefficient =  actual_edges / possible_edges

    print(f"node: {node}")
    print(f"neighbors: {neighbors}, k: {k}")
    print(f"possible_edges: {possible_edges}, actual_edges: {actual_edges}, coefficient: {coefficient}") 
    
    return coefficient

def clustering_coefficient_graph(G):
    clustering_coefficients = [clustering_coefficient(G, node) for node in G.nodes]
    return sum(clustering_coefficients) / len(G.nodes)

def plot_graph(G):
    nx.draw(G, with_labels=True)
    plt.show()


G = nx.Graph()
G.add_edges_from([(1, 2), (1, 3), (1, 4), (2, 3), (3, 4), (4, 5), (4, 6), (5, 6)])

print(clustering_coefficient_graph(G)) 

plot_graph(G)



G1 = nx.Graph()
G1.add_edges_from([(1, 2), (1, 3), (1, 4), (2, 3), (3, 4), (4, 5), (4, 6), (5, 6), (1, 5), (2, 4)])
print(clustering_coefficient_graph(G1)) 
plot_graph(G1)
