import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Create an adjacency matrix (example)
adjacency_matrix = np.array([[0, 1, 1, 0],
                             [1, 0, 1, 1],
                             [1, 1, 0, 1],
                             [0, 1, 1, 0]])

# Create a graph from the adjacency matrix
G = nx.Graph(adjacency_matrix)

# Draw the graph
pos = nx.spring_layout(G)  # Layout algorithm to position the nodes
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10, font_color='black', font_weight='bold')
plt.title("Graph from Adjacency Matrix")
plt.show()