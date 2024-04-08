import random
import matplotlib.pyplot as plt
import pickle
from torch import nn
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
import numpy as np
from multiprocessing import Pool
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Load graph
graph = pickle.load(open('./data/eth_latest_100_block.pickle', 'rb'))
for u, v, data in graph.edges(data=True):
    if 'type' in data:
        del data['type']
    if 'weight' in data:
        del data['weight']


# Node features
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

adj_matrix = torch.tensor(nx.adjacency_matrix(graph, weight='weight').toarray(), dtype=torch.float32)
data = Data(x=node_features, edge_index=adj_matrix.nonzero().t())

# Split data into training and test sets
num_samples = len(node_features)
num_train_samples = int(0.5 * num_samples)  # You can adjust the split ratio
train_indices, test_indices = train_test_split(range(num_samples), train_size=num_train_samples, random_state=42)

# Training data
train_data = Data(x=node_features[train_indices], edge_index=adj_matrix[train_indices][:, train_indices].nonzero().t())

# Test data
test_data = Data(x=node_features[test_indices], edge_index=adj_matrix[test_indices][:, test_indices].nonzero().t())


# Define GCN model
class GCNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Instantiate the GCN models
model = GCNModel(in_channels=7, hidden_channels=20, out_channels=7)

# Define loss and optimizer for both instances
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Temporal Random Walk
def temporal_random_walk(graph, start_node, walk_length):
    walk = [start_node]
    timestamps = []

    for _ in range(walk_length - 1):
        if not graph.has_node(start_node):
            print(f"Node {start_node} not found in graph. Jumping to another node.")
            start_node = random.choice(list(graph.nodes()))
            continue

        neighbors = list(graph.neighbors(start_node))

        if not neighbors:
            # Jump to another random node if there are no neighbors
            start_node = random.choice(list(graph.nodes()))
            continue

        # Sort neighbors based on the timestamp
        neighbors = sorted(neighbors,
                           key=lambda x: graph[start_node][x].get('timestamp', 0) if graph.has_edge(start_node,
                                                                                                    x) else 0,
                           reverse=True)

        # Choose the most recent neighbor, or if timestamps are the same/randomly select one
        start_node = neighbors[0]

        walk.append(start_node)

        # Get the timestamp of the edge or default to 0 if not present
        timestamp = graph[walk[-2]][start_node].get('timestamp', 0) if graph.has_edge(walk[-2], start_node) else 0
        timestamps.append(timestamp)

    return walk, timestamps

# Conduct multiple TRWs and calculate node frequencies
def multiple_temporal_random_walks(graph, num_walks=10, walk_length=100):
    all_walks = []
    for _ in range(num_walks):
        start_node = random.choice(list(graph.nodes()))
        walk, _ = temporal_random_walk(graph, start_node, walk_length)
        all_walks.extend(walk)
    return all_walks

all_walks = multiple_temporal_random_walks(graph, num_walks=10, walk_length=100)

# Initialize the node frequencies
node_freqs = dict.fromkeys(graph.nodes(), 0)
for node in all_walks:
    node_freqs[node] += 1

# Training the model
def train_model(model, data, all_walks_indices, node_freqs, epochs=200):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        relevant_out = out[all_walks_indices]
        relevant_x = data.x[all_walks_indices]
        weights = torch.tensor([node_freqs.get(node, 0) for node in all_walks_indices], dtype=torch.float32)
        loss = (weights * criterion(relevant_out, relevant_x)).mean()
        loss.backward()
        optimizer.step()
    return model

all_walks_indices = [list(graph.nodes()).index(node) for node in all_walks]
trained_model = train_model(model, data, all_walks_indices, node_freqs)

# After training, get embeddings
trained_model.eval()
embeddings = trained_model(data.x, data.edge_index).cpu().detach().numpy()

# Compute anomaly scores
def compute_anomaly_scores(embeddings, node_freqs_values):
    scores = []
    for idx, embedding in enumerate(embeddings):
        mean = np.mean(embedding)
        std = np.std(embedding)
        latest_value = embedding[-1]
        z_score = (latest_value - mean) / std
        weighted_z_score = z_score * node_freqs_values[idx]
        scores.append(weighted_z_score)
    return scores

anomaly_scores = compute_anomaly_scores(embeddings, list(node_freqs.values()))


# Visualization of anomalous nodes
def plot_anomalous_nodes(graph, scores, threshold=2.0):
    anomalous_nodes = [node for idx, node in enumerate(graph.nodes()) if abs(scores[idx]) > threshold]

    pos = nx.spring_layout(graph)
    plt.figure(figsize=(12, 12))

    # Draw all nodes
    nx.draw_networkx_nodes(graph, pos)

    # Highlight anomalous nodes with a different color and size
    nx.draw_networkx_nodes(graph, pos, nodelist=anomalous_nodes, node_color='r', node_size=200)

    # Draw edges
    nx.draw_networkx_edges(graph, pos)

    # Add labels for anomalous nodes
    anomalous_labels = {node: node for node in anomalous_nodes}
    nx.draw_networkx_labels(graph, pos, labels=anomalous_labels, font_color='black')

    plt.title("Anomalous nodes in red")
    plt.show()

    return anomalous_nodes

anomalous_nodes = plot_anomalous_nodes(graph, anomaly_scores)

print(f"Number of anomalous nodes: {len(anomalous_nodes)}")
print(f"Anomalous nodes: {anomalous_nodes}")




import numpy as np

# List of threshold values to experiment with
thresholds = [0.1, 0.5, 1.0, 1.5, 2.0]  # Adjust these threshold values as needed

# Initialize lists to store precision, recall, and F1-score for each threshold
precisions = []
recalls = []
f1_scores = []

# Iterate over different threshold values
for threshold in thresholds:
    # Train the model using only the training data
    trained_model = train_model(model, data, all_walks_indices, node_freqs)

    # Get embeddings for both training and test data
    trained_model.eval()
    train_embeddings = trained_model(train_data.x, train_data.edge_index).cpu().detach().numpy()
    test_embeddings = trained_model(test_data.x, test_data.edge_index).cpu().detach().numpy()

    # Compute anomaly scores for both training and test data
    train_anomaly_scores = compute_anomaly_scores(train_embeddings, list(node_freqs.values()))
    test_anomaly_scores = compute_anomaly_scores(test_embeddings, list(node_freqs.values()))

    # Detect anomalies for training data
    train_detected_anomalies = [idx for idx, score in enumerate(train_anomaly_scores) if abs(score) > threshold]
    print('train_detected_anomalies: ', train_detected_anomalies)
    # Detect anomalies for test data
    test_detected_anomalies = [idx for idx, score in enumerate(test_anomaly_scores) if abs(score) > threshold]
    print('test_detected_anomalies: ', test_detected_anomalies)

    # Calculate precision, recall, and F-score for test data
    test_true_labels = [1 if idx in test_indices else 0 for idx in range(len(node_features))]
    test_detected_labels = [1 if idx in test_detected_anomalies else 0 for idx in range(len(node_features))]

    precision = precision_score(test_true_labels, test_detected_labels)
    recall = recall_score(test_true_labels, test_detected_labels)
    f1 = f1_score(test_true_labels, test_detected_labels)

    # Append the results to the lists
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

# Calculate the average precision, recall, and F1-score
avg_precision = np.mean(precisions)
avg_recall = np.mean(recalls)
avg_f1_score = np.mean(f1_scores)

# Print the average results
print(f"Average Precision: {avg_precision:.3f}")
print(f"Average Recall: {avg_recall:.3f}")
print(f"Average F1-score: {avg_f1_score:.3f}")
