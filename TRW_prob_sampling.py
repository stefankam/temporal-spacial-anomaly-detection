import time
import torch
import torch.nn as nn
import networkx as nx
import pickle
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

# Load your graph data and create the adjacency matrix and node features
# Define the blockchain graph data (adjacency matrix and node features)
import pickle
graph = pickle.load(open('./data/eth_latest_100_block.pickle', 'rb'))
# Modify the node features to include some temporal features
node_features = [
    (
        float(node_data.get('outgoing_tx_count', 0)),
        float(node_data.get('incoming_tx_count', 0)),
        sum(node_data.get('outgoing_value_list', [])),
        sum(node_data.get('incoming_value_list', [])),
        float(node_data.get('activity_rate', 0)),  # Activity rate over a period (needs to be calculated and added to node_data)
        float(node_data.get('change_in_activity', 0)),  # Change in activity over periods (needs to be calculated and added to node_data)
        float(node_data.get('time_since_last', 0)),  # Time since the last transaction or activity (needs to be calculated and added to node_data)
    )
    for node, node_data in graph.nodes(data=True)
]
node_features = torch.tensor(node_features, dtype=torch.float32)

adj_matrix = torch.tensor(nx.adjacency_matrix(graph).toarray(), dtype=torch.float32)

data = Data(x=node_features, edge_index=adj_matrix.nonzero().t())
num_epochs = 1000

## if inductive learning, need subgraph. e.g
from torch_geometric.utils import subgraph
train_mask = torch.rand(data.num_nodes) < 0.5
test_mask = ~train_mask

def random_walk(graph, start_node, walk_length):
    walk = [start_node]
    for _ in range(walk_length - 1):
        neighbors = list(graph.neighbors(start_node))
        if len(neighbors) == 0:
            break
        start_node = neighbors[torch.randint(0, len(neighbors), (1,)).item()]
        walk.append(start_node)
    return walk

def temporal_random_walk(graph, start_node, walk_length):
    walk = [start_node]
    for _ in range(walk_length - 1):
        neighbors = list(graph.neighbors(start_node))
        if len(neighbors) == 0:
            break
        # Sort neighbors based on the timestamp (assuming edge weights represent timestamps)
        #neighbors = sorted(neighbors, key=lambda x: graph[start_node][x]['timestamp'], reverse=True)
        neighbors = sorted(neighbors, key=lambda x: graph[start_node][x].get('timestamp', 0), reverse=True)
        # Choose the most recent neighbor
        start_node = neighbors[0]
        walk.append(start_node)
    return walk

def sample_nodes(graph, walk_function, num_walks=100, walk_length=10):
    walks = []
    nodes = list(graph.nodes())
    for i in range(num_walks):
        start_node = torch.randint(0, len(nodes), (1,)).item()
        walk = walk_function(graph, start_node, walk_length)
        walks.extend(walk)
    return torch.unique(torch.tensor(walks)).tolist()

graph = nx.from_numpy_array(adj_matrix.numpy())

original_sampled_nodes = torch.arange(len(graph.nodes))
# Implement Random Walk-based sampling
rw_sampled_nodes = sample_nodes(graph, random_walk, num_walks=100, walk_length=10)
rw_sampled_nodes = torch.tensor(rw_sampled_nodes)
rw_sampled_adj_matrix = adj_matrix[rw_sampled_nodes][:, rw_sampled_nodes]
rw_sampled_node_features = node_features[rw_sampled_nodes]

# Implement Temporal Random Walk-based sampling
trw_sampled_nodes = sample_nodes(graph, temporal_random_walk, num_walks=100, walk_length=10)
trw_sampled_nodes = torch.tensor(trw_sampled_nodes)
trw_sampled_adj_matrix = adj_matrix[trw_sampled_nodes][:, trw_sampled_nodes]
trw_sampled_node_features = node_features[trw_sampled_nodes]

rw_sampled_data = Data(x=rw_sampled_node_features, edge_index=rw_sampled_adj_matrix.nonzero().t())
trw_sampled_data = Data(x=trw_sampled_node_features, edge_index=trw_sampled_adj_matrix.nonzero().t())
num_epochs = 1000


# Update loss calculation for GCN model training
def train_model(model, optimizer, epochs, sampled_nodes, train_mask, node_features):
    model.train()
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(data.x, data.edge_index)
        loss = criterion(output[sampled_nodes][train_mask[sampled_nodes]], node_features[sampled_nodes][train_mask[sampled_nodes]])
        loss.backward()
        optimizer.step()

def evaluate(model, dataset, sampled_nodes, test_mask, node_features):
    model.eval()
    with torch.no_grad():
        output = model(data.x, data.edge_index)
        pred = output.argmax(dim=1)
        #print("pred")
        #print(pred)
        #print("pred[sampled_nodes][test_mask[sampled_nodes]]")
        #print(pred[sampled_nodes][test_mask[sampled_nodes]])
        #print("node_features.argmax(dim=1)[sampled_nodes][test_mask[sampled_nodes]]")
        #print(node_features.argmax(dim=1)[sampled_nodes][test_mask[sampled_nodes]])
        correct = pred[sampled_nodes][test_mask[sampled_nodes]] == node_features.argmax(dim=1)[sampled_nodes][test_mask[sampled_nodes]]
        acc = int(correct.sum()) / len(correct)
    return acc


# Model and optimizer definitions
gcn_model = GCNConv(in_channels=7, hidden_channels=20, out_channels=7)
sage_model = SAGEConv(in_channels=7, hidden_channels=20, out_channels=7)
gat_model = GATConv(in_channels=7, hidden_channels=20, out_channels=7)

gcn_optimizer = torch.optim.Adam(gcn_model.parameters(), lr=0.01)
sage_optimizer = torch.optim.Adam(sage_model.parameters(), lr=0.01)
gat_optimizer = torch.optim.Adam(gat_model.parameters(), lr=0.01)

# Grouping data, models, and optimizers
datasets = {
    "Original": (data,original_sampled_nodes),
    "rw_sampled": (rw_sampled_data,rw_sampled_nodes),
    "trw_sampled": (trw_sampled_data, trw_sampled_nodes)
}

models = {
    "GCN": (gcn_model, gcn_optimizer),
    "GraphSAGE": (sage_model, sage_optimizer),
    "GAT": (gat_model, gat_optimizer)
}

# Training and evaluating
results = {}
for data_name, (dataset, nodes) in datasets.items():
    print(f"Processing {data_name}")
    print(f"Dataset shape: {dataset.x.shape}")
    print(f"Nodes shape: {len(nodes)}")
    for model_name, (model, optimizer) in models.items():
        start_time = time.time()
        train_model(model, optimizer, num_epochs, nodes, train_mask, node_features)
        duration = time.time() - start_time
        acc = evaluate(model, dataset, nodes, test_mask, node_features)
        results[(data_name, model_name)] = acc
        print(f"Trained {model_name} on {data_name} in {duration:.2f}s with accuracy: {acc:.4f}")

# Plotting the results
labels = list(models.keys())
width = 0.25
x = range(len(labels))

for i, dataset_name in enumerate(datasets.keys()):
    accuracies = [results[(dataset_name, model_name)] for model_name in labels]
    plt.bar([pos + width*i for pos in x], accuracies, width=width, label=dataset_name)

plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Comparison of Model Accuracy Across Datasets')
plt.xticks([pos + width for pos in x], labels)
plt.legend()
plt.tight_layout()
plt.show()
