from gurobipy import *
import copy
import time


def read_in_data(file_name):
    """
    :param file_name: some file name (of a graph) - put in local address of the graph below
    :return: num_nodes, var_list, edge_list, edge_costs

    Read in the file, and store the data
    """
    f = open('/Users/a_gardner/Desktop/Alex Folder/Rice University/Senior Year/Fall 2017/CAAM 574 - Combinatorial'
             ' Optimization/Branch_and_Cut_Solver/%s'
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


def create_matrix(m, num_nodes):
    """
    :param m: Gurobi Model
    :param num_nodes: Number of Nodes
    :return: Weights, variable names

    Creates Dictionaries of dictionaries corresponding to the current connection and the weights
    """

    sol_vars = m.getVars()
    varnames = {}
    weights = {}

    for i in xrange(num_nodes):
        varnames[str(i)] = {}
        weights[str(i)] = {}
        for j in xrange(num_nodes):
            weights[str(i)][str(j)] = -1
            varnames[str(i)][str(j)] = []

    for var in sol_vars:
        value = var.x
        name = var.varName
        node_source, node_sink = name[1:].split('_')

        weights[node_source][node_sink] = value
        weights[node_sink][node_source] = value

        varnames[node_source][node_sink] = [name]
        varnames[node_sink][node_source] = [name]

        weights[node_source][node_source] = -1
        varnames[node_source][node_source] = []

    weights[str(num_nodes - 1)][str(num_nodes - 1)] = -1
    varnames[str(num_nodes - 1)][str(num_nodes - 1)] = []

    return weights, varnames

print read_in_data('ch150.tsp.del')[2]
# comments for days
