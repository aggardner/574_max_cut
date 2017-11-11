from gurobipy import *
import time
import random


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


def is_int(sol_vars):
    """
    :param sol_vars: solution variables (set of gurobi variables)
    :return: Boolean - False if any of the variables are fractional, True otherwise
    """
    for var in sol_vars:
        if 0.0 < var.x < 1.0:
            # print "who's fucking up?", var.varName, var.x
            return False, var
    return True, 'null'


def half_approx_alg(edge_costs):
    """
    :param edge_costs: dictionary of dictionaries {node0:{node1:edge_weight, ..., nodeN:edge_weight},..., nodeN{...}}
    :return: some randomly generated lower bound on the max cut
    """

    # initialize lower bound on max cut and our subset
    max_cut_lb = 0
    a = []

    # find subset of nodes with randomly picking some node at 50% likelihood
    for node in edge_costs:
        rand_num = random.random()
        if rand_num <= .5:
            a.append(node)

    # for each node
    for node in edge_costs:
        # if the node is in our subset
        if node in a:
            # for each of the nodes connected to it
            for incident_edge in edge_costs[node]:
                # if a neighboring node is in the other set, add the edge to the cut
                if incident_edge not in a:
                    max_cut_lb += edge_costs[node][incident_edge]

    return max_cut_lb


def branch_and_cut(file_name):
    """
    :param file_name: Runs the whole function via branch and cut
    :return: The optimal path value and the path itself as a list of tuples with the edge and the coefficient of the
             edge (1.0) for all edges
    """
    # initialize
    iteration = 0
    opt_var = []

    # read in data
    num_nodes, var_list, edge_list, edge_costs = read_in_data(file_name)

    # builds initial model
    m = Model()

    # build things for initial model
    obj = 0
    gurobi_vars = {}
    fractional_vars_seen = {}

    # builds objective function
    for i in xrange(len(var_list)):
        path_variable = m.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=var_list[i])
        gurobi_vars[var_list[i]] = path_variable
        obj += edge_list[i] * path_variable

    # builds node variables
    node_list = {}
    for i in range(num_nodes):
        node_name = 'y%s' % i
        node_variable = m.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=node_name)
        node_list['%s' % i] = node_variable

    m.update()

    # builds initial constraints
    for idx, edge in enumerate(gurobi_vars):
        source, dest = edge.strip('x').split('_')
        partition_constraint_name = 'partition_%s' % idx
        partition_constraint = gurobi_vars[edge] - node_list[source] - node_list[dest]
        m.addConstr(partition_constraint <= 0, partition_constraint_name)

        fancy_constraint_name = 'fancy_%s' % idx
        fancy_constraint = gurobi_vars[edge] + node_list[source] + node_list[dest] - 2
        m.addConstr(fancy_constraint <= 0, fancy_constraint_name)

    # sets objective function
    m.update()
    m.setObjective(obj, GRB.MAXIMIZE)

    # mutes print statements and solves
    m.update()
    m.setParam('OutputFlag', False)
    m.optimize()

    # adds m to queue
    queue = [m]

    # 1/2 Approximation Algorithm to find initial lower bound
    cur_best_solution = half_approx_alg(edge_costs)

    # Begin branch and cut - while there are things in the queue
    while len(queue) > 0:

        iteration += 1

        # remove item from queue
        cur_model = queue.pop(0)

        # THIS NEEDS TO BECOME THE NEW CUT ALGORITHM
        # add viable cuts
        # cur_model = min_cut(cur_model, weights, varnames, num_nodes, gurobi_vars_1, m_temp, a)

        # mute output and solve
        cur_model.setParam('OutputFlag', False)
        cur_model.optimize()

        # take solution variables and check if they're fractional
        sol_vars = cur_model.getVars()
        integer_solution, frac_var = is_int(sol_vars)

        print iteration, cur_model.getAttr("ObjVal"), cur_best_solution, len(queue)
        # keep track of fractional variables seen
        if frac_var not in fractional_vars_seen:
            fractional_vars_seen[frac_var] = 1

        # check to see if we found an integer solution
        if integer_solution:
            # if we did and it's better, replace the current optimal solution
            if cur_model.getAttr("ObjVal") > cur_best_solution:
                cur_best_solution = cur_model.getAttr("ObjVal")
                opt_var = [(v.varName, v.X) for v in cur_model.getVars() if abs(v.x) > 0.0]

        # if it's worse, regardless of whether or not it's integer, do nothing
        elif cur_model.getAttr("ObjVal") <= cur_best_solution:
            continue
        # otherwise, branch and bound
        else:
            # make a copy then add a bounding constraint var >= 1
            m1 = cur_model.copy()
            m1_var_map = {}
            m1_vars = m1.getVars()
            for var in m1_vars:
                m1_var_map[var.varName] = var
            m1.addConstr(m1_var_map[frac_var.varName], '>=', 1)

            # make a copy then add a bounding constraint var <= 0
            m2 = cur_model.copy()
            m2_var_map = {}
            m2_vars = m2.getVars()
            for var in m2_vars:
                m2_var_map[var.varName] = var
            m2.addConstr(m2_var_map[frac_var.varName], '<=', 0)

            # add the models back to the queue and continue
            queue.append(m1)
            queue.append(m2)

    return cur_best_solution, opt_var


file_list = ['gr21.txt']
best_sols = []
run_times = []
for filename in file_list:
    print filename
    start_time = time.time()
    best_sol, opt_vars = branch_and_cut(filename)
    print best_sol, '\n', opt_vars, '\n'
    graph_time = time.time() - start_time
    best_sols.append(best_sol)
    run_times.append(graph_time)
    print '\n'

print file_list
print best_sols
print run_times
