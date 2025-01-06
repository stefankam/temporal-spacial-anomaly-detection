import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

# Lists to hold complexities
gcn_complexities = []
trw_complexities = []
# Function to extract subgraphs from a directed graph, ensuring subgraph is weakly connected
def get_subgraph(original_graph, target_size):
    # Use weakly connected components for a directed graph
    largest_cc = max(nx.weakly_connected_components(original_graph), key=len)
    large_subgraph = original_graph.subgraph(largest_cc).copy()
    nodes_sample = random.sample(large_subgraph.nodes(), min(target_size, len(large_subgraph)))
    subgraph = original_graph.subgraph(nodes_sample).copy()
    return subgraph

# Sizes of the subgraphs to test
subgraph_sizes = [500, 1000, 1500, 2000, 2500]  # Adjust based on the original graph size

# Storing complexities for plotting
combined_complexities = []
F = features_per_node  # number of features
H = 20  # number of hidden units in GCN
L = 10  # number of layers in GCN

# Iterate over different subgraph sizes
for size in subgraph_sizes:
    subgraph = get_subgraph(graph, size)
    num_nodes = len(subgraph)
    num_edges = subgraph.number_of_edges()

    Delta = max(len(list(graph.neighbors(n))) for n in graph.nodes())  # Maximum degree

    # Compute complexities
    trw_complexity = num_nodes * 100
    gcn_complexity = L * (num_edges + num_nodes * F * H)

    gcn_complexities.append(gcn_complexity)
    trw_complexities.append(trw_complexity)

    # Combined complexity
    combined_complexity = trw_complexity + gcn_complexity
    combined_complexities.append(combined_complexity)

# Plotting the complexities
plt.figure(figsize=(10, 5))
plt.plot(subgraph_sizes, gcn_complexities, label='GCN Complexity', marker='o')
plt.plot(subgraph_sizes, trw_complexities, label='TRW Complexity', marker='x')
plt.plot(subgraph_sizes, combined_complexities, label='TRW-GCN Combined Complexity', marker='o')
plt.xlabel('Number of Nodes in Subgraph')
plt.ylabel('Combined Complexity')
plt.title('TRW-GCN Complexity Scaling with Graph Size')
plt.legend()
plt.grid(True)
plt.show()