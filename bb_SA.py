from gurobipy import *
import copy
import time

from graph import *
def read_in_data(file_name):
    """
    :param file_name: some file name (of a graph) - put in local address of the graph below
    :return: num_nodes, var_list, edge_list, edge_costs

    Read in the file, and store the data
    """
    m=Model()
    f = open('./Inputs/%s'
             % file_name, 'r')

    text_data = f.readlines()
    # First line of file
    num_nodes, num_edges = text_data[0].split()
    num_nodes, num_edges = int(num_nodes), int(num_edges)

    node_list ={}
    edge_list = {}
    edge_costs = {}

    for i in range(0, num_nodes):
        edge_costs[str(i)] = {}

        node_name= 'x%s' % (i)
        node_variable=m.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=node_name)
        node_list['%s' % (i)]=node_variable

        for j in range(0, num_nodes):
            edge_costs[str(i)][str(j)] = -1
    obj=0
    for i in range(1, num_edges + 1):
        line = text_data[i].split()
        if len(line) == 1:
            break
        node1, node2, edge_weight = line
        
        #Edge costs
        edge_costs[str(node1)][str(node2)] = int(edge_weight)
        edge_costs[str(node2)][str(node1)] = int(edge_weight)

        edge_name = 'z%s_%s' % (node1, node2)
        path_variable=m.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=edge_name)
        edge_list['%s_%s' % (node1, node2)]=path_variable


        obj+=int(edge_weight)*path_variable
    m.update()
    m.setObjective(obj, GRB.MAXIMIZE)

    for idx, edge in enumerate(edge_list):
        source, dest= edge.strip('z').split('_')

        node_list[(source)]

        partition_constraint_name='partition_%s' %idx
        partition_constraint=edge_list[edge]-node_list[source]-node_list[dest]
        m.addConstr(partition_constraint<=0, partition_constraint_name)

        fancy_constraint_name='fancy_%s' %idx
        fancy_constraint=edge_list[edge]+node_list[source]+node_list[dest]-2
        m.addConstr(fancy_constraint<=0, fancy_constraint_name)
   

    m.update()
    m.write('bb.lp')
    m.optimize()
    m.write('bb.sol')
    g=Graph(edge_list, node_list, edge_costs)
   


read_in_data('gr21.txt')
