import numpy as np
# import networkx as nx
# import matplotlib.pyplot as plt

def edge_list_to_adjacency_matrix(edge_list, num_nodes):
    adjacency_matrix = [[0] * num_nodes for _ in range(num_nodes)]
    for edge in edge_list:
        node1, node2 = edge
        adjacency_matrix[node1 - 1][node2 - 1] += 1
        adjacency_matrix[node2 - 1][node1 - 1] += 1
    return adjacency_matrix

def parse_nodes_and_edges(line):
    line = line.split(',')
    line = [ element.replace("m", "").replace("n", "").replace("=", "") for element in line ]
    line = [ int(element) for element in line ]
    return line

def parse_edge_list_with_weights(line):
    line = line.split(' ')
    line = [ element.replace("{", " ").replace("}", " ").replace(",", " ") for element in line ]
    line = [ element.strip().split(' ') for element in line ]
    
    for sublist in line:
        sublist[:] = [int(element) for element in sublist]
    # Change into tuple for easier access in for loop
    line = [tuple(sublist) for sublist in line]

    return line

def build_weighted_graph(nodes_edges, edge_list_with_weights):
    nodes_num = nodes_edges[0]

    adjacency_matrix = [[0] * nodes_num for _ in range(nodes_num)]

    for node1, node2, weight in edge_list_with_weights:
        adjacency_matrix[node1][node2] = weight
        adjacency_matrix[node2][node1] = weight

    return np.array(adjacency_matrix)

# def plot_wieghted_graph(adjacency_matrix):
#     G = nx.Graph()

#     for i in range(len(adjacency_matrix)):
#         for j in range(i, len(adjacency_matrix[i])):
#             if adjacency_matrix[i][j] != 0:
#                 G.add_edge(i, j, weight=adjacency_matrix[i][j])

#     pos = nx.spring_layout(G) 
#     labels = nx.get_edge_attributes(G, 'weight')
#     nx.draw(G, pos, with_labels=True, node_size=300, node_color='lightblue')
#     nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
#     plt.show()

def check_connected_graph(graph_matrix):
    def dfs(node):
        visited[node] = True
        for neighbor in range(len(graph_matrix)):
            if graph_matrix[node][neighbor] > 0 and not visited[neighbor]:
                dfs(neighbor)

    n = len(graph_matrix)
    visited = [False] * n

    # Start DFS from the first node
    dfs(0)

    # Returns true or false depending if the graph is connected or not
    return all(visited)

def prims_algorithm(nodes_and_edges, edge_list):
    visited = set()
    nodes_added_num = 0
    spanning_tree = []
    nodes_num = nodes_and_edges[0]

    final_weight = 0
    sorted_edge_list = sorted(edge_list, key=lambda item: item[2])
    while nodes_added_num < nodes_num - 1:
        visited.add(sorted_edge_list[0][0])
        for edge in sorted_edge_list:
            node1, node2, weight = edge
            if (node1 not in visited and node2 in visited) or (node1 in visited and node2 not in visited):
                visited.add(node1)
                visited.add(node2)
                spanning_tree.append(edge)
                nodes_added_num += 1
                break
    for element in spanning_tree:
        final_weight += element[2]
    print('Weights of the graph:', final_weight)
    
    
if __name__ == "__main__":
    with(open("./input.txt", "r")) as file:
        lines = file.readlines()
        test_cases = int(lines[0])
        lines = lines[1:]

        for line1, line2 in zip(lines[::2], lines[1::2]):
            matrix = build_weighted_graph(parse_nodes_and_edges(line1.strip()), parse_edge_list_with_weights(line2.strip()))
            is_connected = check_connected_graph(matrix)
            if is_connected:
                prims_algorithm(parse_nodes_and_edges(line1.strip()),parse_edge_list_with_weights(line2.strip()))
            else:
                print("Graf is not connected - no spanning tree")

        