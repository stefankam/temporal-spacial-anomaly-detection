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
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import numpy as np
from multiprocessing import Pool
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score, recall_score, f1_score  # Add these imports
from sklearn.model_selection import train_test_split  # Add this import
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

# Load graph
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

adj_matrix = torch.tensor(nx.adjacency_matrix(graph, weight='weight').toarray(), dtype=torch.float32)
data = Data(x=node_features, edge_index=adj_matrix.nonzero().t())

# Split the data into train and test sets
train_ratio = 0.5  # You can adjust this ratio as needed
num_samples = len(node_features)
num_train_samples = int(train_ratio * num_samples)

train_indices, test_indices = train_test_split(range(num_samples), train_size=num_train_samples, random_state=42)


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
model_with_trw = GCNModel(in_channels=7, hidden_channels=20, out_channels=7)
model_without_trw = GCNModel(in_channels=7, hidden_channels=20, out_channels=7)

# Define loss and optimizer for both instances
optimizer_with_trw = torch.optim.Adam(model_with_trw.parameters(), lr=0.01)
optimizer_without_trw = torch.optim.Adam(model_without_trw.parameters(), lr=0.01)
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

# Train GCN using a subgraph from the sampled nodes
def train_subgraph(nodes, model, optimizer):
    # Assuming you have the entire graph loaded as `graph`
    node_to_idx = {node: idx for idx, node in enumerate(graph.nodes())}
    idx_to_node = {idx: node for node, idx in node_to_idx.items()}
    # Convert your list of nodes to indices
    node_indices = [node_to_idx[node] for node in nodes]
    # create a subgraph
    subgraph = torch.tensor(node_indices, dtype=torch.long)
    # train
    model.train()
    optimizer.zero_grad()
    output = model(data.x, data.edge_index)
    loss = criterion(output[subgraph], node_features[subgraph])
    loss.backward()
    optimizer.step()
    return loss.item()


# Train GCN using nodes sampled from TRWs or without sampling
def train_and_get_anomalies(all_walks, sampling=True, model=None, optimizer=None):

    if sampling:  # Train GCN using nodes sampled from TRWs
        for epoch in range(10):
            random.shuffle(all_walks)
            for i in range(0, len(all_walks), walk_length):
                batch_nodes = all_walks[i: i + walk_length]
                train_subgraph(batch_nodes, model, optimizer)
    else:  # Train GCN using all nodes
        for epoch in range(10):
            batch_nodes = list(graph.nodes())
            train_subgraph(batch_nodes, model, optimizer)

    # Obtain GCN embeddings
    embeddings = model(data.x, data.edge_index).cpu().detach().numpy()
    embeddings[np.isnan(embeddings)] = 0

    # Anomaly detection using dbscan
    dbscan = DBSCAN(eps=0.5, min_samples=5).fit(embeddings)
    labels = dbscan.labels_
    gcn_detected_nodes_dbscan = set(np.where(labels == -1)[0])
    #anomalous_nodes = np.where(labels == -1)[0]
    #node_features = data.x.cpu().detach().numpy()  # Assuming your node features are stored in data.x
    #analyze_anomalous_nodes(anomalous_nodes, model, node_features)

    # Anomaly detection using One-Class SVM
    ocsvm = OneClassSVM(kernel='rbf', nu=0.1, gamma=0.01).fit(embeddings)
    predictions = ocsvm.predict(embeddings)
    gcn_detected_nodes_svm = set(np.where(predictions == -1)[0])
    #anomalous_nodes = np.where(labels == -1)[0]
    #node_features = data.x.cpu().detach().numpy()  # Assuming your node features are stored in data.x
    #analyze_anomalous_nodes(anomalous_nodes, model, node_features)

    # Anomaly detection using Isolation Forest
    iso_forest = IsolationForest(contamination=0.05).fit(embeddings)
    predictions_iso_forest = iso_forest.predict(embeddings)
    gcn_detected_nodes_isoforest = set(np.where(predictions_iso_forest == -1)[0])
    #anomalous_nodes = np.where(labels == -1)[0]
    #node_features = data.x.cpu().detach().numpy()  # Assuming your node features are stored in data.x
    #analyze_anomalous_nodes(anomalous_nodes, model, node_features)

    # Anomaly detection using LOF
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
    predictions_lof = lof.fit_predict(embeddings)
    gcn_detected_nodes_lof = set(np.where(predictions_lof == -1)[0])
    anomalous_nodes = np.where(predictions_lof == -1)[0]
    node_features = data.x.cpu().detach().numpy()  # Assuming your node features are stored in data.x
    analyze_anomalous_nodes(anomalous_nodes, model, node_features)

    return gcn_detected_nodes_dbscan, gcn_detected_nodes_svm, gcn_detected_nodes_isoforest, gcn_detected_nodes_lof


def analyze_anomalous_nodes(anomalous_nodes, model, node_features):
    # Extract node features of anomalous nodes
    anomalous_features = node_features[anomalous_nodes]

    scaler = StandardScaler()
    if anomalous_features.shape[0] > 0:
        scaled_anomalous_features = scaler.fit_transform(anomalous_features)

        # Analyze feature distribution
        n_features = scaled_anomalous_features.shape[1]
        colors = plt.cm.viridis(np.linspace(0, 1, n_features))  # Choose a colormap to select colors from

        plt.figure(figsize=(12, 8))
        for idx in range(n_features):
            plt.hist(scaled_anomalous_features[:, idx], bins=30, color=colors[idx], alpha=0.5, label=f'Feature {idx}')

        plt.xlabel("Feature Value (normalized)", fontsize=20)
        plt.ylabel("Frequency", fontsize=20)
        plt.xlim(-10, 10)  # Set the x-axis limits
        plt.legend(loc='upper right', prop={'size': 20})
        plt.title("Feature Distribution of Anomalous Nodes", fontsize=20)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tight_layout()
        plt.show()


        # Clustering anomalous nodes to identify patterns
        n_clusters = 3  # Or any appropriate number based on your domain knowledge
        kmeans = KMeans(n_clusters=n_clusters).fit(anomalous_features)
        cluster_labels = kmeans.labels_

        # For each cluster, print the mean feature values
        for cluster in range(n_clusters):
            print(f"Mean Feature Values for Cluster {cluster}:", np.mean(anomalous_features[cluster_labels == cluster], axis=0))


# Perform TRWs to sample nodes
num_walks = 10
walk_length = 100

# Train GCN using nodes sampled from TRWs
all_walks = multiple_temporal_random_walks(graph, num_walks, walk_length)

# Get anomalies with TRWs sampling
(gcn_detected_nodes_dbscan_with_trw,
 gcn_detected_nodes_svm_with_trw,
 gcn_detected_nodes_isoforest_with_trw,
 gcn_detected_nodes_lof_with_trw) = train_and_get_anomalies(all_walks, sampling=True, model=model_with_trw, optimizer=optimizer_with_trw)

# Get anomalies without TRWs (using all nodes)
(gcn_detected_nodes_dbscan_without_trw,
 gcn_detected_nodes_svm_without_trw,
 gcn_detected_nodes_isoforest_without_trw,
 gcn_detected_nodes_lof_without_trw) = train_and_get_anomalies(all_walks, sampling=False, model=model_without_trw, optimizer=optimizer_without_trw)

# Visualization
methods = ['GCN_DBSCAN_with_TRW', 'GCN_SVM_with_TRW', 'GCN_ISOFOREST_with_TRW', 'GCN_LOF_with_TRW',
           'GCN_DBSCAN_without_TRW', 'GCN_SVM_without_TRW', 'GCN_ISOFOREST_without_TRW', 'GCN_LOF_without_TRW']


node_counts = [
    len(gcn_detected_nodes_dbscan_with_trw),
    len(gcn_detected_nodes_svm_with_trw),
    len(gcn_detected_nodes_isoforest_with_trw),
    len(gcn_detected_nodes_lof_with_trw),
    len(gcn_detected_nodes_dbscan_without_trw),
    len(gcn_detected_nodes_svm_without_trw),
    len(gcn_detected_nodes_isoforest_without_trw),
    len(gcn_detected_nodes_lof_without_trw)
]

plt.figure(figsize=(15, 7))
bars = plt.bar(methods, node_counts, color=['green', 'purple', 'blue', 'red', 'yellow', 'cyan', 'orange', 'pink'])

# Increase the font size for the bar labels
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 5, round(yval, 2), ha='center', va='bottom', fontsize=12)  # Font size for numbers on bars

# Increase the font size for the title, x-axis labels, and y-axis labels
plt.title('Comparison of Anomaly Detection using GCN with TRW and without TRW', fontsize=20)  # Increase title font size
plt.xlabel('Methods', fontsize=16)  # Increase x-axis label font size
plt.ylabel('Number of Nodes', fontsize=16)  # Increase y-axis label font size

# Increase the font size for the x-tick labels (method names)
plt.xticks(rotation=45, fontsize=16)  # Increase font size for x-tick labels

plt.tight_layout()
plt.show()





# Create a dictionary to store the detected nodes for each method without TRW
detected_nodes_without_trw = {
    'DBSCAN_without_TRW': gcn_detected_nodes_dbscan_without_trw,
    'SVM_without_TRW': gcn_detected_nodes_svm_without_trw,
    'IsolationForest_without_TRW': gcn_detected_nodes_isoforest_without_trw,
    'LOF_without_TRW': gcn_detected_nodes_lof_without_trw,
}

# Calculate precision, recall, and F-score for each method without TRW
scores_without_trw = {}

for method, detected in detected_nodes_without_trw.items():
    true_labels_without_trw = [1 if i in test_indices else 0 for i in range(len(node_features))]
    detected_labels_without_trw = [1 if i in detected else 0 for i in range(len(node_features))]

    precision_without_trw = precision_score(true_labels_without_trw, detected_labels_without_trw)
    recall_without_trw = recall_score(true_labels_without_trw, detected_labels_without_trw)
    f1_without_trw = f1_score(true_labels_without_trw, detected_labels_without_trw)

    scores_without_trw[method] = {'Precision': precision_without_trw, 'Recall': recall_without_trw,
                                  'F-score': f1_without_trw}

# Print the precision, recall, and F-score for each method without TRW
for method, metrics in scores_without_trw.items():
    print(f'{method}:')
    print(f'Precision: {metrics["Precision"]}')
    print(f'Recall: {metrics["Recall"]}')
    print(f'F-score: {metrics["F-score"]}')


# Create a dictionary to store the detected nodes for each method with TRW
detected_nodes_with_trw = {
    'DBSCAN_with_TRW': gcn_detected_nodes_dbscan_with_trw,
    'SVM_with_TRW': gcn_detected_nodes_svm_with_trw,
    'IsolationForest_with_TRW': gcn_detected_nodes_isoforest_with_trw,
    'LOF_with_TRW': gcn_detected_nodes_lof_with_trw,
}

# Calculate precision, recall, and F-score for each method with TRW
scores_with_trw = {}

for method, detected in detected_nodes_with_trw.items():
    true_labels_with_trw = [1 if i in test_indices else 0 for i in range(len(node_features))]
    detected_labels_with_trw = [1 if i in detected else 0 for i in range(len(node_features))]

    precision_with_trw = precision_score(true_labels_with_trw, detected_labels_with_trw)
    recall_with_trw = recall_score(true_labels_with_trw, detected_labels_with_trw)
    f1_with_trw = f1_score(true_labels_with_trw, detected_labels_with_trw)

    scores_with_trw[method] = {'Precision': precision_with_trw, 'Recall': recall_with_trw, 'F-score': f1_with_trw}

# Print the precision, recall, and F-score for each method with TRW
for method, metrics in scores_with_trw.items():
    print(f'{method}:')
    print(f'Precision: {metrics["Precision"]}')
    print(f'Recall: {metrics["Recall"]}')
    print(f'F-score: {metrics["F-score"]}')

