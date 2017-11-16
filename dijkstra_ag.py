import copy


def read_in_data(file_name):
    """
    :param file_name: some file name (of a graph) - put in local address of the graph below
    :return: num_nodes, var_list, edge_list, edge_costs

    Read in the file, and store the data
    """
    f = open('./Inputs/%s'
             % file_name, 'r')

    text_data = f.readlines()
    # First line of file
    num_nodes, num_edges = text_data[0].split()

    num_nodes, num_edges = int(num_nodes), int(num_edges)

    var_list = []
    edge_list = []
    edge_costs = {}

    for i in range(0, num_nodes):
        edge_costs[str(i)] = {}

        for j in range(0, num_nodes):
            edge_costs[str(i)][str(j)] = 0

    for i in range(1, num_edges + 1):
        line = text_data[i].split()
        if len(line) == 1:
            break
        node1, node2, edge_weight = line

        edge_costs[str(node1)][str(node2)] = int(edge_weight)
        edge_costs[str(node2)][str(node1)] = int(edge_weight)

        edge_list.append(int(edge_weight))
        var_name = 'x%s_%s' % (node1, node2)
        var_list.append(var_name)

    return num_nodes, var_list, edge_list, edge_costs


def build_pseudograph(gurobi_edge_vals):
    """
    :param gurobi_edge_vals:
    :return pseudo_graph:
    """
    original_nodes = gurobi_edge_vals.keys()

    temp = gurobi_edge_vals.copy()
    for node in original_nodes:
        new_node = '%sP' % node
        temp[new_node] = gurobi_edge_vals[node]

    p_nodes = temp.copy()

    for node in temp:
        if node in original_nodes:
            p_nodes.pop(node)

    o_nodes = copy.deepcopy(gurobi_edge_vals)
    for node in gurobi_edge_vals:
        for element in gurobi_edge_vals[node]:
            o_nodes[node][element + 'P'] = o_nodes[node].pop(element)

    pseudograph = dict(p_nodes.items() + o_nodes.items())

    for node in pseudograph:
        if ('P' in node) and (node.strip('P') in pseudograph[node]):
                pseudograph[node].pop(node.strip('P'))
        if ('P' not in node) and (node+'P' in pseudograph[node]):
            pseudograph[node].pop(node + 'P')

    return pseudograph


def dijkstra_ag(graph, start_node):
    """
    :param graph: An undirected graph
    :param start_node: A node in the graph from which the search starts
    :return: Two dictionaries in a tuple: The first is the every node's distance from the start node
            The second is the number of shortest paths from the node in question to the starting node

    Performs dijkstra's algorithm on the graph starting at the node start_node to determine how far
    each node is from the starting node, and the number of shortest paths the node has to the starting
    node.
    """
    distance = {}
    # distance maps the node's distance from the start node

    paths = {}
    # paths initial

    nodes = graph.keys()

    for node in graph:
        # initially assume that each node is infinitely far away from the starting node
        distance[node] = float('inf')
        # initialize initial paths to be blank - haven't been visited
        paths[node] = []

    # initializes starting node to have a path that begins at itself
    paths[start_node] = [start_node]

    distance[start_node] = 0
    # distance from the start_node is initialized at zero

    # Initialize the search queue
    queue = nodes

    # Explore all connected nodes
    while queue:
        # while the queue is not empty

        # takes the item w/ min distance from start still in the queue

        node = min(distance.viewkeys() & queue, key=distance.get)

        queue.pop(queue.index(node))

        for neighbor in graph[node]:
            # for each neighbor of the node

            alt = distance[node] + graph[node][neighbor]

            if alt < distance[neighbor]:
                distance[neighbor] = alt
                paths[neighbor] = [x for x in paths[node]]
                paths[neighbor].append(neighbor)

    return paths

file_name = 'gr21.txt'
num_nodes, var_list, edge_list, edge_costs = read_in_data(file_name)
pseudograph = build_pseudograph(edge_costs)
visited, path = dijkstra_ag(pseudograph, '0')
