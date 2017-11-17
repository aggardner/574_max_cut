from gurobipy import *
import time
import random
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


def add_odd_cuts(model, graph, edge_gurobi_map):
    """
    :param model:
    :param graph:
    :param edge_gurobi_map:
    :return:
    """

    add_cut_indicator = True

    model_constrs = []
    while add_cut_indicator:

        # Make sure we are sending the right things to the function! - BUILD THIS!!!
        gurobi_edge_vals = get_current_weights(model, graph)

        for var in model.getVars():
            name = var.varName
            edge_gurobi_map[name] = var

        added_list = []
        
        pseudograph = build_pseudograph(gurobi_edge_vals)
        for node in graph:
            constraint, violated_flag = generate_odd_cut_constr(pseudograph, node, edge_gurobi_map)
            constr_set=set()
            for i in range(0, constraint[0].size()):
                constr_set.add(str(constraint[0].getVar(i).varName.strip(' ')))

            if violated_flag and constr_set not in model_constrs:
                model.addConstr(constraint[0], constraint[1], constraint[2], name='OddCut%d'%count)
                model_con=set()
                for i in range(0, constraint[0].size()):
                    model_con.add(str(constraint[0].getVar(i).varName.strip(' ')))
                model_constrs.append(model_con)

                model.optimize()
                add_flag = True
                added_list.append(add_flag)

        if sum(added_list) == 0:
            add_cut_indicator = False

    return model


def get_current_weights(m, graph):
    """
    :param m: Gurobi Model
    :param num_nodes: Number of Nodes
    :return: Weights, variable names

    Creates Dictionaries of dictionaries corresponding to the current connection and the weights
    """

    sol_vars = m.getVars()

    for node1 in graph:
        for node2 in graph[node1]:
            graph[node1][node2] = 0

    for var in sol_vars:
        if 'x' in var.varName: # edge variable
            value = var.x
            name = var.varName
            node_source, node_sink = name[1:].split('_')

            graph[node_source][node_sink] = 1 - value
            graph[node_sink][node_source] = 1 - value

    return graph


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


def generate_odd_cut_constr(psuedograph, node, edge_vars):
    """
    :param psuedograph:
    :param node:
    :param model:
    :return constraint:
    """

    paths = dijkstra(psuedograph, node)
    path = paths[node + 'P']
    
    # strip the P from the node name
    actual_path = []
    for node in path:
        if node[-1] == 'P':
            actual_path.append(node[:-1])
        else:
            actual_path.append(node)

    # get the cycle from that path
    cycle = find_odd_cycle(actual_path)

    # build the actual constraint
    rhs = len(cycle) - 2
    sense = '<='

    cycle_sum = 0

    # loop to build constraint out of the model variables
    lhs = LinExpr(0)
    for i in range(0, len(cycle)-1):
        source = str(min(int(cycle[i]), int(cycle[i+1])))
        sink = str(max(int(cycle[i]), int(cycle[i+1])))
        var = 'x' + source + '_' + sink
        lhs += edge_vars[var]

        cycle_sum +=  1-edge_vars[var].x

    # builds constraint
    constraint = [lhs, sense, rhs]

    return constraint, cycle_sum < 1


def dijkstra(graph, start_node):
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
                paths[neighbor] = paths[node] + [neighbor]

    return paths


def find_odd_cycle(path):
    """
    :param path:
    :return:
    """
    min_dist = len(path)
    sub_index = [0, len(path)-1]
    seen = {}
    index = 0
    for node in path:
        if node not in seen:
            seen[node] = [index]
        else:
            seen[node].append(index)
            diff = abs(seen[node][0] - seen[node][1])
            if (diff % 2 == 1) and (diff < min_dist):
                sub_index = seen[node]
                min_dist = diff
        index += 1

    low = min(sub_index)
    high = max(sub_index)
    cycle = path[low: high + 1]

    return cycle


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
    edge_vars = {}
    fractional_vars_seen = {}

    # builds objective function
    for i in xrange(len(var_list)):

        path_variable = m.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=var_list[i])
        #print var_list[i]
        edge_vars[var_list[i]] = path_variable
        obj += edge_list[i] * path_variable
    # builds node variables
    node_vars = {}
    for i in range(num_nodes):
        node_name = 'y%s' % i
        node_variable = m.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=node_name)
        node_vars['%s' % i] = node_variable

    m.update()

    # builds initial constraints
    for idx, edge in enumerate(edge_vars):
        source, dest = edge.strip('x').split('_')
        partition_constraint_name = 'partition_%s' % idx
        partition_constraint = edge_vars[edge] - node_vars[source] - node_vars[dest]
        m.addConstr(partition_constraint <= 0, partition_constraint_name)

        fancy_constraint_name = 'fancy_%s' % idx
        fancy_constraint = edge_vars[edge] + node_vars[source] + node_vars[dest] - 2
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

        # add viable cuts
        cur_model = add_odd_cuts(cur_model, edge_costs, edge_vars)

        cur_model.update()

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

            m1.setParam('OutputFlag', False)
            m1.optimize()

            sol_vars_1 = m1.getVars()
            integer_solution, _ = is_int(sol_vars_1)

            if integer_solution:
                if m1.getAttr("ObjVal") > cur_best_solution:
                    cur_best_solution = m1.getAttr("ObjVal")
                    # m1.write('%sx_curr_best_sol.lp' % file_name)
                    opt_var = [(v.varName, v.X) for v in m1.getVars() if v.x > 0.0]
            else:
                if m1.getAttr("ObjVal") > cur_best_solution:
                    queue.append(m1)

            m2.setParam('OutputFlag', False)
            m2.optimize()

            sol_vars_2 = m2.getVars()
            integer_solution, _ = is_int(sol_vars_2)

            if integer_solution:
                if m2.getAttr("ObjVal") > cur_best_solution:
                    cur_best_solution = m2.getAttr("ObjVal")
                    # m2.write('%sx_curr_best_sol.lp' % file_name)
                    opt_var = [(v.varName, v.X) for v in m2.getVars() if abs(v.x) > 0.0]
            else:
                if m2.getAttr("ObjVal") > cur_best_solution:
                    queue.append(m2)

    return cur_best_solution, opt_var


file_list = ['d657.tsp.del']
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
