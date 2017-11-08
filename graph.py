class Graph:

	def __init__(self, edge_variables, node_variables, edge_costs):
		self.edge_variables=edge_variables
		self.node_variables=node_variables
		self.edge_costs=edge_costs
		self.num_nodes=len(node_variables)
		self.num_edges=len(edge_variables)