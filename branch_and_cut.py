from gurobipy import *
import time
import random
import copy


def read_in_data(file_name):
    """
    :param file_name: some file name (of a graph) - put in local address of the graph below
    :return: num_nodes, var_list, edge_list, edge_costs

    Reads in a graph instance and traverses through all the lines to create node and edge variables and keeping track of weights.
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

    #Method to check if a given LP solution is a Integer solution, within tolerance. 
    """
    for var in sol_vars:
        if 0.000000000001 < var.x < .99999999999:
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


def add_odd_cuts(model, graph, edge_gurobi_map,model_constrs):
    """
    :param model: Gurobi model
    :param graph: current graph
    :param edge_gurobi_map: maps edge variables names to gurobi variable counterparts
    :model_constrs; set of current odd cuts
    :return:
    """

    add_cut_indicator = True

    #Algorithm: Continously build odd cuts until we are no longer able to
    while add_cut_indicator:

        gurobi_edge_vals = get_current_weights(model, graph)
        #Map our the gurobi variable names to actual variables. This is needed because the reference of each gurobi model is being changed, 
        #Thus we need to keep track of these references.
        for var in model.getVars():
            name = var.varName
            edge_gurobi_map[name] = var

        added_list = []
        #Retrieve our new modified grah
        pseudograph = build_pseudograph(gurobi_edge_vals)
        #Build a series of odd cuts per node in the graph
        for node in graph:
            constraint, violated_flag = generate_odd_cut_constr(pseudograph, node, edge_gurobi_map)
            constr_set = set()
            for i in range(0, constraint[0].size()):
                constr_set.add(str(constraint[0].getVar(i).varName.strip(' ')))
            #Only add an odd cut if it is feasible, i.e the inequality sum(edges)<=S-1 was violated. If so, then this cut will then be useful
            #Also check to see we have not alreay added the cut
            if violated_flag and constr_set not in model_constrs:

                model.addConstr(constraint[0], constraint[1], constraint[2], name='OddCut')
                #Add the new cut into our current set of Odd cuts so we don't ever add it back again
                model_con=frozenset(str(constraint[0].getVar(i).varName.strip(' ')) for i in range(0, constraint[0].size()))
                model_constrs.add(model_con)

                model.optimize()
                #Add flag means we have added an odd cut. If we have never added an odd cut in our graph, add_flag would remain false and hence the loop is exited
                add_flag = True
                added_list.append(add_flag)
                #CHeck to see if we are integer
                sol_vars = model.getVars()
                integer_solution, _ = is_int(sol_vars)
                if integer_solution:
                    add_cut_indicator = False

        if sum(added_list) == 0:
            add_cut_indicator = False

    return model, model_constrs


def get_current_weights(m, graph):
    """
    :param m: Gurobi Model
    :param graph: Current Graph
    :return:
    """

    sol_vars = m.getVars()
    #Given the current graph, modify the edgeweights to be 1-value so we can use this graph to determine what cuts to make
    for node1 in graph:
        for node2 in graph[node1]:
            graph[node1][node2] = 0

    for var in sol_vars:
        if 'x' in var.varName:  # edge variable
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

    #We initially have an nxn graph, but now extend it to 2n x 2n graph. Each pseudonode is detonated as "P" along with its node number.
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
    :param edge_vars:
    :return:
    """
    #To add the odd cuts we first must find the shortest odd path which is obtained form dijkstra's
    #After this we then extract the shortest odd cycle contained in this path

    paths = dijkstra(psuedograph, node)
    path = paths[node + 'P']

    # strip the P from the node name as we want the actual node
    actual_path = []
    for node in path:
        if node[-1] == 'P':
            actual_path.append(node[:-1])
        else:
            actual_path.append(node)

    # get the cycle from that path
    cycle = find_odd_cycle(actual_path)

    # build the actual constraint. It is minus 2 since cylce has k nodes, and therefore k-1 edges
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

        cycle_sum += 1 - edge_vars[var].x

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

    #Given the odd path from dijkstras, identify the shortest odd cycle in this path
    min_dist = len(path)
    sub_index = [0, len(path)-1]
    seen = {}
    index = 0
    #Traverse through our list, and keep track of the indices. Anytime we see a node previously encountered,
    #Calculate the distance between the nodes and use this as our minimum distnace. Update this accordingly as we traverse through the rest of the nodes
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
    #Extract the cycle
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
    #Main driver 
    # initialize
    iteration = 0
    opt_var = []

    # read in data
    num_nodes, var_list, edge_list, edge_costs = read_in_data(file_name)

    # builds initial model
    m = Model()

    # Gurobi model preparation for initial model
    obj = 0
    edge_vars = {}
    fractional_vars_seen = {}

    # builds objective function
    for i in xrange(len(var_list)):

        path_variable = m.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=var_list[i])
        # print var_list[i]
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
    queue = [(m, set())]

    # 1/2 Approximation Algorithm to find initial lower bound
    cur_best_solution = half_approx_alg(edge_costs)

    # Begin branch and cut - while there are things in the queue

    #Our queue consists of an array of tuples. Each tuple contains a gurobi model, and constraint set.
    #We keep a runnign tab of odd cuts in a set assoicated to each model in a tuple. This allows us to check if an odd cut has already been added in a given model

    #In this segment, we pop off a model and do the following
    #1) Add all the odd cuts 
    #2)Check to see if we have an integer solution
        #-If so, is the objective value better than what we already have? Update. Nothing added to the queue
        #If not, is the objective value worse than what we already have? Don't add anything to the queue, this tree is pointless
        #IF not, but the objective value is better than what we have? Time to branch and cut
    while len(queue) > 0:

        iteration += 1

        # remove item from queue
        cur_model, current_cuts= queue.pop(0)

        # add viable cuts
        cur_model, total_cuts = add_odd_cuts(cur_model, edge_costs, edge_vars, current_cuts)

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

            #Are these models, m1 and m2 even worth pursuing?. Optimize and check for the same conditions as before

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
                    queue.append((m1, total_cuts))

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
                    queue.append((m2, total_cuts))

    return cur_best_solution, opt_var


# file_list = os.listdir('Inputs')
# file_list.pop(0)

# file_list = ['att48.txt', 'hk48.txt', 'ulysses22.txt', 'gr21.txt']
# file_list = ['d1291.tsp.del', 'd657.tsp.del']


#Driver instances
file_list = ['a280.tsp.del', 'bier127.tsp.del', 'ch130.tsp.del', 'ch150.tsp.del', 'd198.tsp.del']


best_sols = []
run_times = []
#Loops througha  given set of files and runs branch and cut algorithm on each instances. The times and solutions are then recorded and then written as an output.txt file
for filename in file_list:
    print filename
    start_time = time.time()

    best_sol, opt_vars = branch_and_cut(filename)
    graph_time = time.time() - start_time
    print best_sol, '\n', opt_vars, '\n', graph_time
    best_sols.append(best_sol)
    run_times.append(graph_time)
    print '\n'

#Writes out solutions to output.txt
target=open("output.txt", 'w')
for i in xrange(len(file_list)):
    data='File: %s Time: %s Final Solution: %s\n' %(file_list[i], run_times[i], best_sols[i])
    target.write(data)

print file_list
print best_sols
print run_times

target.close()