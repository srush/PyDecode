import decoding.interfaces.hypergraph_pb2 as hg
import networkx as nx
import matplotlib.pyplot as plt

class Edge:

	def __init__(self, id, label, tail_nodes):
		self.id = id
		self.label = label
		self.tail_node_ids = tail_nodes
	def print_edge(self):
		print self.id
		print self.label
		for n in self.tail_node_ids:
			print n


class Node:

	def __init__(self, id, label, edges = []):
		self.id = id
		self.label = label
		self.edges = edges
	def add_edge(self, edge):
		self.edges.append(edge)
	def print_node(self):
		print self.id,
		print self.label,
		for e in self.edges:
			print e


class Hypergraph:

	def __init__(self):
		self.edge_count = 0
		self.nodes = []
		self.edges = []
	def add_edge(self, label, head_node, tail_nodes):
		e = Edge(self.edge_count, label, tail_nodes)
		self.edge_count += 1
		self.nodes[head_node].add_edge(e)
		print "*", e
		print "*", head_node
		print "*", self.nodes[head_node]
	def add_node(self, label, edges = None):
		if edges is None:
			edges = []
		n = Node(len(self.nodes), label, edges)
		self.nodes.append(n)
		return len(self.nodes) - 1
	def add_root(self, label, edges = None):
		if edges is None:
			edges = []
		id = self.add_node(label, edges)
		self.set_root(id)
		return id
	def get_node_label(self, id) :
		return self.nodes[id].label
	def set_root(self, root_id):
		self.root = root_id
	def number_nodes(self):
		return len(self.nodes)
	def number_edges(self):
		return self.edge_count
	def to_protobuf(self):
		hypergraph_proto = hg.Hypergraph()
		hypergraph_proto.root = self.root
		for node in self.nodes:
			node_proto = hypergraph_proto.node.add()
			node_proto.id = node.id
			node_proto.label = str(node.label)
			for edge in node.edges:
				edge_proto = node_proto.edge.add()
				edge_proto.id = edge.id
				edge_proto.label = str(edge.label)
				for tail_node_id in edge.tail_node_ids:
					edge_proto.tail_node_ids.append(tail_node_id)
		return hypergraph_proto
	def to_networkx(self):
		graph = nx.Graph()
		for node in self.nodes:
			head_node = node.id
			graph.add_node(head_node)
			for edge in node.edges:
				if len(edge.tail_node_ids) == 1 :
					graph.add_edge(head_node, edge.tail_node_ids[0])
				else :
					artificial_node = str(edge.id) + "*"
					graph.add_node(artificial_node)
					graph.add_edge(head_node, artificial_node)
					for tail_node_id in edge.tail_node_ids :
						graph.add_edge(artificial_node, tail_node_id)
		#nx.draw_networkx(graph, pos=nx.spring_layout(graph))
		##nx.draw_networkx_labels(graph, pos=nx.spring_layout(graph))
		#plt.savefig("path.png")
		return graph
	def print_hypergraph(self):
		print "NODES"
		for node in self.nodes:
			node.print_node()
		print "EDGES"
		for edge in self.edges:
			edge.print_edge()
node = Node(0,"D",[])
node.add_edge(1)
node.print_node()
print "---"
hypergraph = Hypergraph()
hypergraph.add_node(0,[])
hypergraph.print_hypergraph()
print "---"
