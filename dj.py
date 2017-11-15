import sys, time
def dj(matrix, node_to_use):
	num_nodes=len(matrix)
	node_to_retreive=str(node_to_use)+'P'
	distance_to_nodes=[10000000]*num_nodes #Make super large
	path_to_nodes={}
	for node in matrix.keys():
		path_to_nodes[node]=[]
	nodes_used=set()
	#print matrix
	distance_to_nodes[node_to_use]=0

	cur_node=0
	dist_so_far=0

	nodes_used.add(node_to_use)
 	#print adjacent_matrix[node_to_use]
	while len(nodes_used)<num_nodes:
		min_so_far=100000 #make super large
		for cur_node in nodes_used:
			for node, weight in adjacent_matrix[cur_node].iteritems():
					if weight>0 and node not in nodes_used:
						cur_dist=weight+distance_to_nodes[cur_node]
						#print cur_node,cur_dist, min_so_far, node, weight
						if weight>0 and cur_dist<min_so_far:
							#print "got a min at node", node
							
							min_so_far=cur_dist
							min_node=node
							path_to_nodes[min_node]=path_to_nodes[cur_node]+[cur_node]

		
		distance_to_nodes[min_node]=min_so_far
		#print distance_to_nodes, time.sleep(4), min_node, cur_node
		nodes_used.add(min_node)

		#print  min_node, min_so_far, distance_to_nodes 

	#print 'Final distances', distance_to_nodes

	print 'Final paths', path_to_nodes
	#Uncomment me, need the +so that we are officially a "cycle"
	#return path_to_nodes[node_to_retreive]+[node_to_retreive]
if __name__ == "__main__":
	f=open('geeksDJ.txt')
	first_line=f.readline().strip(' ').split(' ')
	num_nodes, num_edges=int(first_line[0]), int(first_line[1])
	adjacent_matrix={}
	#http://www.geeksforgeeks.org/greedy-algorithms-set-6-dijkstras-shortest-path-algorithm/
	for i in xrange(num_nodes):
		adjacent_matrix[i]={}
	#adjacent_matrix=[ [0]*num_nodes for i in xrange(num_nodes)]
	for line in f.readlines():
		line=line.strip(' ').split(' ')
		if len(line)<3:
			break
		node1, node2, edge_weight=int(line[0]), int(line[1]), int(line[2])
		adjacent_matrix[node1][node2]=edge_weight
		adjacent_matrix[node2][node1]=edge_weight
	dj(adjacent_matrix,0)