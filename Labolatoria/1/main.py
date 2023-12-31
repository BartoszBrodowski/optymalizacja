import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def update_adjacency_matrix(graph):
    adjacency_matrix = nx.to_numpy_array(graph, dtype=int)
    return adjacency_matrix

def create_graph(matrix):
    return nx.MultiGraph(matrix)

def create_directed_graph(matrix):
    return nx.MultiDiGraph(matrix)

def plot_graph(graph):
    position = nx.spring_layout(graph)
    nx.draw(graph, position, with_labels=True)
    adjacency_matrix = update_adjacency_matrix(graph)
    def get_edge_list(adjacency_matrix):
        edge_list = []
        for i in range(len(adjacency_matrix)):
            for j in range(len(adjacency_matrix)):
                if adjacency_matrix[i][j] != 0:
                    for k in range(adjacency_matrix[i][j]):
                        edge_list.append((i, j))
        return edge_list
    plt.show()
    print(get_edge_list(adjacency_matrix))

def add_node(graph):
    try:
        node_label = int(len(graph.nodes()))
        graph.add_node(node_label)
        update_adjacency_matrix(graph)
    except ValueError:
        print('Invalid input. Please enter a valid integer.')

def remove_node(graph):
    node = input("Which node to remove?: ")
    try:
        int_node = int(node)
        graph.remove_node(int_node)
        update_adjacency_matrix(graph)
    except ValueError:
        print('Invalid input. Please enter a valid integer.')

def add_edge(graph):
    print("Connect nodes (if the graph is directed the connection goes from first to second node):")
    try:
        node1 = int(input("Node 1: "))
        node2 = int(input("Node 2: "))
        if node1 in graph.nodes() and node2 in graph.nodes():
            graph.add_edge(node1, node2)
            update_adjacency_matrix(graph)
            print(f"Edge added between Node {node1} and Node {node2}.")
        else:
            print("One or both of the nodes do not exist in the graph. Please make sure both nodes exist before adding an edge.")
    except ValueError:
        print('Invalid input. Please enter a valid integer.')

def remove_edge(graph):
    print("Which nodes to disconnect?: ")
    try:
        node1 = int(input("Node 1: "))
        node2 = int(input("Node 2: "))
        graph.remove_edge(node1, node2)
        update_adjacency_matrix(graph)
    except ValueError:
        print('Invalid input. Please enter a valid integer.')

def show_degrees(graph):
    degrees = dict(graph.degree())
    for node, degree in degrees.items():
        print(f"Node {node}: Degree {degree}")

def show_node_degree(graph):
    try:
        node = int(input("Which node to show degree?: "))
        degree = graph.degree(node)
        if isinstance(graph, nx.MultiDiGraph):
            incoming_degree = graph.in_degree(node)
            outgoing_degree = graph.out_degree(node)
            print(f"Outcoming Degree: {outgoing_degree}, Incoming Degree: {incoming_degree}")
            return
        print(degree)
        print(f"Node {node}: Degree {degree}")
    except ValueError:
        print('Invalid input. Please enter a valid integer.')

def show_max_min_degree(graph):
    degrees = graph.degree()
    max_degree = max(degrees, key=lambda x: x[1])
    min_degree = min(degrees, key=lambda x: x[1])
    print(f"Maximum Degree: Node {max_degree[0]} with degree {max_degree[1]}")
    print(f"Minimum Degree: Node {min_degree[0]} with degree {min_degree[1]}")

def show_even_odd_degree_nodes(graph):
    degrees = graph.degree()
    even_nodes = []
    odd_nodes = []
    for node, degree in degrees:
        if degree % 2 == 0:
            even_nodes.append(node)
        else:
            odd_nodes.append(node)
    print(f"Even nodes amount: {len(even_nodes)}, Odd nodes amount: {len(odd_nodes)}")

def show_node_degrees_nonascending(graph):
    degrees = graph.degree()
    sorted_degrees = sorted(degrees, key=lambda x: x[1], reverse=True)
    print(sorted_degrees)

def print_pretty_adjacency_matrix(adjacency_matrix):
    num_nodes = len(adjacency_matrix)

    print("   ", end="")
    for i in range(num_nodes):
        print(f"Node {i}  ", end="")
    print("\n")

    for i in range(num_nodes):
        print(f"Node {i} ", end="")
        for j in range(num_nodes):
            print(f"{int(adjacency_matrix[i, j]):>6} ", end="")
        print("\n")

def edge_list_to_adjacency_matrix(graph_type, edge_list, num_nodes):
    adjacency_matrix = [[0] * num_nodes for _ in range(num_nodes)]
    print(edge_list, num_nodes)
    if graph_type == "1":
        for edge in edge_list:
            node1, node2 = edge
            adjacency_matrix[node1 - 1][node2 - 1] += 1
            adjacency_matrix[node2 - 1][node1 - 1] += 1
    elif graph_type == "2":
        for edge in edge_list:
            node1, node2 = edge
            adjacency_matrix[node1 - 1][node2 - 1] += 1

    print(adjacency_matrix)
    return adjacency_matrix

def find_c3_cycle_naive(graph):
    matrix = nx.to_numpy_array(graph, dtype=int)
    n = len(matrix)
    print(matrix)
    for node1 in range(n):
        for node2 in range(n):
            # check first edge, if exists continue
            if matrix[node1][node2] == 1:
                # looking for third neighbour
                for node3 in range(n):
                    # 1 2 == 1, 0 2 == 1
                    if matrix[node2][node3] == 1 and matrix[node1][node3] == 1:
                        # Znaleziono cykl C3
                        print(f'Found C3 cycle, nodes: ', node1, node2, node3)
                        return [node1, node2, node3] 
                    
def find_c3_cycle_matrix_multiplication(graph):
    array = nx.to_numpy_array(graph, dtype=int)
    matrix = np.matrix(array)
    matrix_squared = matrix * matrix
    labels_list = list(graph.nodes())
    temp = None
    n = len(matrix)
    for i in range(n):
        for j in range(n):
            if matrix[i, j] > 0 and matrix_squared[i, j] > 0:
                temp = j
        for node in range(n):
            if matrix[i, node] > 0 and matrix_squared[temp, node] > 0:
                print(f'Found C3 cycle, nodes: , {labels_list[i]}, {labels_list[temp]}, {labels_list[node]}')    
                return [i, temp, node]
            
def add_row_to_dfs_tree(dfs_tree):
    new_row = [0] * len(dfs_tree[0])

def show_dfs_tree(graph, node, visited, dfs_tree):
    def dfs(graph, node, visited, dfs_tree):
        dfs_tree_matrix = [[]]
        matrix = nx.to_numpy_array(graph, dtype=int)
        def find_neighbours(node):
            neighbours = []
            row = matrix[node]
            for index, connection in enumerate(row):
                if connection == 1:
                    neighbours.append(index)
            return neighbours
        
        if node not in visited:
            visited.add(node)
            neighbours = find_neighbours(node)
            for neighbor in neighbours:
                if neighbor not in visited:
                    dfs_tree.add_edge(node, neighbor)
                    dfs_tree_matrix
                    dfs(graph, neighbor, visited, dfs_tree)

    dfs(graph, node, visited, dfs_tree)

    pos = nx.spring_layout(dfs_tree)
    nx.draw(dfs_tree, pos, with_labels=True, node_size=500, node_color='lightgreen')
    print(update_adjacency_matrix(dfs_tree))
    plt.show()
    
running = True


if __name__ == "__main__":
    with(open("./input.txt", "r")) as file:
        edge_list = np.loadtxt(file, dtype=int)
        print(edge_list)
        print("""Choose a graph type:
                1. Undirected
                2. Directed
                """)
        graph_type = input("Enter your choice: ")
        num_nodes = max(max(edge) for edge in edge_list)
        adjacency_matrix = edge_list_to_adjacency_matrix(graph_type, edge_list, num_nodes)
        adjacency_matrix = np.array(adjacency_matrix)
    if graph_type == "1":
        print(adjacency_matrix)
        graph = create_graph(adjacency_matrix)
    elif graph_type == "2":
        graph = create_directed_graph(adjacency_matrix)
    while running:
        print("""
            Choose an option:
            1. Add node
            2. Remove node
            3. Add edge
            4. Remove edge
            5. Show degrees
            6. Show node degree
            7. Show max and min degree nodes
            8. Show even and odd degree nodes
            9. Show node degrees in non-ascending order
            10. Plot the graph
            11. Print adjacency matrix
            12. Check if graph has C3 subgraph (naive)
            13. Check if graph has C3 subgraph (matrix multiplication)
            14. DFS
            15. Exit
        """)

        option = input("Enter your choice: ")

        if option == "1":
            for neighbour in graph.neighbors(0):
                print(type(neighbour))
            add_node(graph)
        elif option == "2":
            remove_node(graph)
        elif option == "3":
            add_edge(graph)
        elif option == "4":
            remove_edge(graph)
        elif option == "5":
            show_degrees(graph)
        elif option == "6":
            show_node_degree(graph)
        elif option == "7":
            show_max_min_degree(graph)
        elif option == "8":
            show_even_odd_degree_nodes(graph)
        elif option == "9":
            show_node_degrees_nonascending(graph)
        elif option == "10":
            plot_graph(graph)
        elif option == '11':
            adjacency_matrix = nx.to_numpy_array(graph, dtype=int)
            print_pretty_adjacency_matrix(adjacency_matrix)
        elif option == '12':
            find_c3_cycle_naive(graph)
        elif option == '13':
            find_c3_cycle_matrix_multiplication(graph)
        elif option == '14':
            show_dfs_tree(graph, 1, set(), nx.MultiGraph())
        elif option == "15":
            print("Thank you for using this program. Have a great day!")
            break
        else:
            print("Invalid option. Please choose a valid option.")