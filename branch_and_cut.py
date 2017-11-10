from gurobipy import *
import copy
import time


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
    node_list={}
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
        #print edge_weight


        edge_costs[str(node1)][str(node2)] = int(edge_weight)
        edge_costs[str(node2)][str(node1)] = int(edge_weight)
        #print edge_costs[str(node2)][str(node1)], edge_costs[str(node2)][str(node1)]
        #time.sleep(2)
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
        if 'x' in name:
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


def is_int(sol_vars):
    """
    :param sol_vars: solution variables (set of gurobi variables)
    :return: Boolean - False if any of the variables are fractional, True otherwise
    """
    for var in sol_vars:
        if 0.0 < var.x < 1.0:
            #print "whose fucking up?", var.varName, var.x
            return False, var
    return True, 'null'


def branch_and_cut(file_name):
    """
    :param file_name: Runs the whole function via branch and cut
    :return: The optimal path value and the path itself as a list of tuples with the edge and the coefficient of the
             edge (1.0) for all edges
    """

    iteration = 0
    opt_var = []

    num_nodes, var_list, edge_list, edge_costs = read_in_data(file_name)
    # print "Loading Data for %s" % file_name

    # builds initial model
    m = Model()

    obj = 0
    gurobi_vars = {}
    fractional_vars_seen={}

    for i in xrange(len(var_list)):
        path_variable = m.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=var_list[i])
        gurobi_vars[var_list[i]] = path_variable
        obj += edge_list[i] * path_variable

    node_list={}

    for i in range(num_nodes):
        node_name='y%s' % (i)
        node_varable= m.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=node_name)
        node_list['%s' % (i)]=node_varable

    m.update()

    for idx, edge in enumerate(gurobi_vars):
        source, dest = edge.strip('x').split('_')
        partition_constraint_name='partition_%s' %idx
        partition_constraint=gurobi_vars[edge]-node_list[source]-node_list[dest]
        m.addConstr(partition_constraint<=0, partition_constraint_name)

        fancy_constraint_name='fancy_%s' %idx
        fancy_constraint=gurobi_vars[edge]+node_list[source]+node_list[dest]-2
        m.addConstr(fancy_constraint<=0, fancy_constraint_name)

    m.update()
    m.setObjective(obj, GRB.MAXIMIZE)

    # for node1 in xrange(num_nodes):
    #     constraint = LinExpr(0)
    #     for node2 in xrange(num_nodes):
    #         if node1 != node2:
    #             if node1 < node2:
    #                 edge_variable = 'x%i_%i' % (node1, node2)
    #             else:
    #                 edge_variable = 'x%i_%i' % (node2, node1)
    #             gurobi_var = gurobi_vars[edge_variable]
    #             constraint += gurobi_var
    #     constraint_name = 'initial_constraint%i' % node1
    #     m.addConstr(constraint == 2, constraint_name)

    m.update()
    # m.write("%s_initial.lp" % file_name)
    # print("Model Initialized")
    print('we in here')
    m.setParam('OutputFlag', False)

    m.optimize()

    print m.getAttr("ObjVal")
    # begin branch or cut
    queue = [m]

    # THIS NEEDS TO BECOME 1/2 APPROXIMATION ALGORITHM
    # cur_best_solution = nearest_neighbor(edge_costs)
    cur_best_solution = -10000000000

    while len(queue) > 0:
        m_temp = Model()

        gurobi_vars_1 = {}
        obj2 = 0

        for i in xrange(len(var_list)):
            path_var_1 = m_temp.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=var_list[i])
            gurobi_vars_1[var_list[i]] = path_var_1
            obj2 += edge_list[i] * path_var_1

        m_temp.update()
        m_temp.setObjective(obj2)

        a = '1'
        iteration += 1

        cur_model = queue.pop(0)

        

        # print 'defined'
        # THIS NEEDS TO BECOME THE NEW CUT ALGORITHM
        # cur_model = min_cut(cur_model, weights, varnames, num_nodes, gurobi_vars_1, m_temp, a)

        # print 'cutted'
        # cur_model.write('%s_cut.lp' % file_name)
        cur_model.setParam('OutputFlag', False)
        cur_model.optimize()
        #cur_model.write('Modexp%s.lp'%iteration)
        #cur_model.write('Modex%s.sol'%iteration)
        weights, varnames = create_matrix(cur_model, num_nodes)
        #print cur_model.getAttr("ObjVal"), "SHIIITTTTT"
        # print 'solved'
        # ALl Viable cuts added, resolved. Check if integer

        sol_vars = cur_model.getVars()
        integer_solution, frac_var = is_int(sol_vars)
        #print "val "
        #print sol_vars[0], cur_model.STATUS
        print iteration, cur_model.getAttr("ObjVal"), cur_best_solution, len(queue)
        if frac_var not in fractional_vars_seen:
            fractional_vars_seen[frac_var]=1
        else:
            pass
           # print "This shouldn't happen?", frac_var
            #time.sleep(100)
        if integer_solution:
            print "SNAPPP" 

        #print "is it integer?", integer_solution, len(queue)
        #time.sleep(4)
        if integer_solution:
            # print 'integer solution found', cur_model.getAttr("ObjVal")
            if cur_model.getAttr("ObjVal") > cur_best_solution:
                cur_best_solution = cur_model.getAttr("ObjVal")
                # cur_model.write('%sx_curr_best_sol.lp' % file_name)
                opt_var = [(v.varName, v.X) for v in cur_model.getVars() if abs(v.x) > 0.0]
        else:
            if cur_model.getAttr("ObjVal") > cur_best_solution:

                m1 = cur_model.copy()

                m1_var_map = {}
                m1_vars = m1.getVars()
                for var in m1_vars:
                    m1_var_map[var.varName] = var
                #print "Fractional variable", frac_var.varName
                m1.addConstr(m1_var_map[frac_var.varName], '>=', 1)


                m2 = cur_model.copy()

                m2_var_map = {}
                m2_vars = m2.getVars()
                for var in m2_vars:
                    m2_var_map[var.varName] = var

                m2.addConstr(m2_var_map[frac_var.varName], '<=', 0)

                queue.append(m1)
                queue.append(m2)
                #print "IDentifying constriants"
                #m1.write("Model1_%i.lp"% iteration)
                #m2.write("Model2_%i.lp" % iteration)
        #         #time.sleep(5)
        #         # m1.write('%sx_B0.lp' % file_name)
        #         m1.setParam('OutputFlag', False)
        #         m1.optimize()

        #         sol_vars_1 = m1.getVars()
        #         integer_solution, _ = is_int(sol_vars_1)

        #         if integer_solution:
        #             # print 'integer solution found', m1.getAttr("ObjVal")
        #             if m1.getAttr("ObjVal") > cur_best_solution:
        #                 cur_best_solution = m1.getAttr("ObjVal")
        #                 # m1.write('%sx_curr_best_sol.lp' % file_name)
        #                 opt_var = [(v.varName, v.X) for v in m1.getVars() if v.x > 0.0]
        #         else:
        #             if m1.getAttr("ObjVal") > cur_best_solution:
        #                 queue.append(m1)

        #         # m2.write('%sx_B1.lp' % file_name)

        #         m2.setParam('OutputFlag', False)
        #         m2.optimize()

        #         sol_vars_2 = m2.getVars()
        #         integer_solution, _ = is_int(sol_vars_2)

        #         if integer_solution:
        #             # print 'integer solution found', m2.getAttr("ObjVal")
        #             if m2.getAttr("ObjVal") > cur_best_solution:
        #                 cur_best_solution = m2.getAttr("ObjVal")
        #                 # m2.write('%sx_curr_best_sol.lp' % file_name)
        #                 opt_var = [(v.varName, v.X) for v in m2.getVars() if abs(v.x) > 0.0]
        #         else:
        #             if m2.getAttr("ObjVal") > cur_best_solution:
        #                 queue.append(m2)

        # #print 'best integer =', cur_best_solution, 'len(queue) =', len(queue)

    return cur_best_solution, opt_var



file_name='ch150.tsp.del'
# comment

read_in_data(file_name)


file_list = ['gr21.txt']
best_sols = []
run_times = []
for file in file_list:
    print file
    start_time = time.time()
    best_sol, opt_var = branch_and_cut(file)
    print best_sol, '\n', opt_var, '\n'
    graph_time = time.time() - start_time
    best_sols.append(best_sol)
    run_times.append(graph_time)
    print '\n'

print file_list
print best_sols
print run_times

