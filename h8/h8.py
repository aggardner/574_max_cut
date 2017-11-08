from gurobipy import *

def maxcut(graph):

	edge_map={0:"a", 1:"b", 2:"c", 3:"g", 4:"e", 5:"f", 6: "h", 7:"i"}
	edges_seen={}
	count=0
	m=Model()

	node_vars={}
	edge_vars={}
	for i in xrange(len(graph)):
		var_name='x%s' % (i)
		node_variable=m.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY, name=var_name)
		node_vars[i]=node_variable
	m.update()
	#print node_vars
	obj=0
	for i in xrange(len(graph)):
		for j in xrange(len(graph)):
			if i!=j and graph[i][j]>0 and "%s_%s" % (i,j) not in edges_seen and "%s_%s" % (j,i) not in edges_seen:
				edges_seen["%s_%s" % (j,i)]=1
				edges_seen["%s_%s" % (i,j)]=1

 				count+=1
 				var_name='z%s_%s' % (i, j)
 				path_variable=m.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY, name=var_name)
 				edge_vars["%s_%s" % (i,j)]=path_variable
 				weight=graph[i][j]
 				obj+=weight*path_variable
 	m.setObjective(obj, GRB.MAXIMIZE)

 	for idx, edge in enumerate(edge_vars):
 		source, dest= edge.split('_')
 		#print source, dest
 		partition_constraint_name='partition_%s' %idx
 		partition_constraint=edge_vars[edge]-node_vars[int(source)]-node_vars[int(dest)]
 		m.addConstr(partition_constraint<=0, partition_constraint_name)

 		fancy_constraint_name='fancy_%s' %idx
 		fancy_constraint=edge_vars[edge]+node_vars[int(source)]+node_vars[int(dest)]-2
 		m.addConstr(fancy_constraint<=0, fancy_constraint_name)
 	m.update()
 	m.write('AYYYYYY_ourLP.lp')
 	#print count
 	m.optimize()
 	m.write("AYYYY_oursol.sol")
 	variables=m.getVars()
 	print "----Solution Output----"
 	for var in variables:
 		if "z" in var.varName and var.x==1.0:
 			edge_name=var.varName.strip('z')
 			source, dest= edge_name.split('_')
 			source_letter, dest_letter=edge_map[int(source)], edge_map[int(dest)]
 			print source_letter, dest_letter

graph= [\
#a b c g e f h i
[0,9,0,0,0,0,10,5],
[9,0,11,0,7,3,7,2],
[0,11,0,0,0,10,6,0],
[0,0,0,0,0,2,4,1],
[0,7,0,0,0,8,0,10],
[0,0,10,2,8,0,8,6],
[10,7,6,4,0,8,0,5],
[5,2,0,1,10,6,5,0]]
maxcut(graph)