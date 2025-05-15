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
graph = pickle.load(open('data/eth_latest_100_block_20240506.pickle', 'rb'))
# Modify the node features to include some temporal features
node_features = [
    (
        #float(node_data.get('outgoing_tx_count', 0)),
        #float(node_data.get('incoming_tx_count', 0)),
        #sum(node_data.get('outgoing_value_list', [])),
        #sum(node_data.get('incoming_value_list', [])),
        #float(node_data.get('activity_rate', 0)),  # Activity rate over a period (needs to be calculated and added to node_data)
        #float(node_data.get('change_in_activity', 0)),  # Change in activity over periods (needs to be calculated and added to node_data)
        #float(node_data.get('time_since_last', 0)),  # Time since the last transaction or activity (needs to be calculated and added to node_data)
        # Variance calculations
        float(node_data.get('incoming_value_variance', 0)),
        float(node_data.get('outgoing_value_variance', 0)),
        # Calculate activity metrics
        float(node_data.get('activity_rate', 0)),
        # Activity rate over a period (needs to be calculated and added 3to node_data)
        float(node_data.get('change_in_activity', 0)),
        # Change in activity over periods (needs to be calculated and added to node_data)
        float(node_data.get('time_since_last', 0)),
        # Time since the last transaction or activity (needs to be calculated and added to node_data)
        # Calculate total transaction volume
        float(node_data.get('tx_volume', 0)),
        # Identify addresses with frequent and large transfers
        float(node_data.get('frequent_large_transfers', 0)),
        # Additional features for MEV detection
        float(node_data.get('gas_price', 0)),
        float(node_data.get('token_swaps', 0)),
        float(node_data.get('smart_contract_interactions', 0))
    )
    for node, node_data in graph.nodes(data=True)
]
node_features = torch.tensor(node_features, dtype=torch.float32)

# Convert the weighted adjacency matrix to a dense matrix
adj_matrix_dense = nx.to_numpy_array(graph, weight='weight')
# Convert the dense matrix to a PyTorch tensor
adj_matrix = torch.tensor(adj_matrix_dense, dtype=torch.float32)
data = Data(x=node_features, edge_index=adj_matrix.nonzero().t())

# Split the data into train and test sets
train_ratio = 0.8  # You can adjust this ratio as needed
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
model_with_trw = GCNModel(in_channels=10, hidden_channels=20, out_channels=10)
model_without_trw = GCNModel(in_channels=10, hidden_channels=20, out_channels=10)

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




def train_and_get_anomalies2(all_walks, dbscan_params, svm_params, iso_params, lof_params,
    sampling=True, model=None, optimizer=None):

    model.train()
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

    gcn_detected_nodes_dbscan = set(np.where(DBSCAN(**dbscan_params).fit(embeddings).labels_ == -1)[0])
    gcn_detected_nodes_svm = set(np.where(OneClassSVM(**svm_params).fit(embeddings).predict(embeddings) == -1)[0])
    gcn_detected_nodes_isoforest = set(np.where(IsolationForest(**iso_params).fit(embeddings).predict(embeddings) == -1)[0])
    gcn_detected_nodes_lof = set(np.where(LocalOutlierFactor(**lof_params).fit_predict(embeddings) == -1)[0])

    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
    predictions_lof = lof.fit_predict(embeddings)
    anomalous_nodes = np.where(predictions_lof == -1)[0]
    node_features = data.x.cpu().detach().numpy()  # Assuming your node features are stored in data.x
    analyze_anomalous_nodes(anomalous_nodes, model, node_features)

    return gcn_detected_nodes_dbscan, gcn_detected_nodes_svm, gcn_detected_nodes_isoforest, gcn_detected_nodes_lof


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

methods = [
    'GCN_DBSCAN_with_TRW', 'GCN_SVM_with_TRW', 'GCN_ISOFOREST_with_TRW', 'GCN_LOF_with_TRW',
    'GCN_DBSCAN_without_TRW', 'GCN_SVM_without_TRW', 'GCN_ISOFOREST_without_TRW', 'GCN_LOF_without_TRW'
]

num_runs = 1
node_counts_runs = []
precision_runs = {method: [] for method in methods}
recall_runs = {method: [] for method in methods}
f1_runs = {method: [] for method in methods}

for run in range(num_runs):
    print(f"Run {run + 1}/{num_runs}")
    # Set seed
    seed = 42 + run
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # Instantiate the GCN models
    model_with_trw = GCNModel(in_channels=10, hidden_channels=20, out_channels=10)
    model_without_trw = GCNModel(in_channels=10, hidden_channels=20, out_channels=10)

    # Define loss and optimizer for both instances
    optimizer_with_trw = torch.optim.Adam(model_with_trw.parameters(), lr=0.01)
    optimizer_without_trw = torch.optim.Adam(model_without_trw.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Define varying parameters (example: randomize or use different presets)
    eps = 0.4 + 0.05 * (run % 3)  # cycles through 0.4, 0.5, 0.6
    contamination = 0.03 + 0.05 * (run % 4)  # cycles through 0.03, 0.04, ...
    nu = 0.08 + 0.005 * (run % 3)
    gamma = 0.005 + 0.001 * (run % 3)

    # Perform TRWs to sample nodes
    num_walks = 10
    walk_length = 1000

    # Generate walks
    all_walks = multiple_temporal_random_walks(graph, num_walks, walk_length)


    gcn_dbscan_trw, gcn_svm_trw, gcn_iso_trw, gcn_lof_trw = train_and_get_anomalies2(
            all_walks,
            sampling=True,
            model=model_with_trw,
            optimizer=optimizer_with_trw,
            dbscan_params={"eps": eps, "min_samples": 5},
            svm_params={"nu": nu, "gamma": gamma},
            iso_params={"contamination": contamination},
            lof_params={"n_neighbors": 20, "contamination": contamination}
        )
#     train_and_get_anomalies(
#     all_walks, sampling=True, model=model_with_trw, optimizer=optimizer_with_trw))

    gcn_dbscan_no_trw, gcn_svm_no_trw, gcn_iso_no_trw, gcn_lof_no_trw = train_and_get_anomalies2(
            all_walks,
            sampling=False,
            model=model_without_trw,
            optimizer=optimizer_without_trw,
            dbscan_params={"eps": eps, "min_samples": 5},
            svm_params={"nu": nu, "gamma": gamma},
            iso_params={"contamination": contamination},
            lof_params={"n_neighbors": 20, "contamination": contamination}
        )

    current_counts = [
        len(gcn_dbscan_trw), len(gcn_svm_trw), len(gcn_iso_trw), len(gcn_lof_trw),
        len(gcn_dbscan_no_trw), len(gcn_svm_no_trw), len(gcn_iso_no_trw), len(gcn_lof_no_trw)
    ]
    node_counts_runs.append(current_counts)

    detected_dict = {
        'GCN_DBSCAN_with_TRW': gcn_dbscan_trw,
        'GCN_SVM_with_TRW': gcn_svm_trw,
        'GCN_ISOFOREST_with_TRW': gcn_iso_trw,
        'GCN_LOF_with_TRW': gcn_lof_trw,
        'GCN_DBSCAN_without_TRW': gcn_dbscan_no_trw,
        'GCN_SVM_without_TRW': gcn_svm_no_trw,
        'GCN_ISOFOREST_without_TRW': gcn_iso_no_trw,
        'GCN_LOF_without_TRW': gcn_lof_no_trw
    }

    true_labels = [1 if i in train_indices else 0 for i in range(len(node_features))]

    for method, detected in detected_dict.items():
        predicted_labels = [1 if i in detected else 0 for i in range(len(node_features))]
        precision_runs[method].append(precision_score(true_labels, predicted_labels))
        recall_runs[method].append(recall_score(true_labels, predicted_labels))
        f1_runs[method].append(f1_score(true_labels, predicted_labels))

# --- Averaged Metrics ---
mean_node_counts = np.mean(node_counts_runs, axis=0)
std_node_counts = np.std(node_counts_runs, axis=0)

# --- Visualization ---
plt.figure(figsize=(15, 7))
bars = plt.bar(methods, mean_node_counts, yerr=std_node_counts, capsize=5,
               color=['green', 'purple', 'blue', 'red', 'yellow', 'cyan', 'orange', 'pink'])

for bar, mean in zip(bars, mean_node_counts):
    plt.text(bar.get_x() + bar.get_width() / 2, mean + 2, round(mean, 2),
             ha='center', va='bottom', fontsize=12)

plt.title(f'Average Anomaly Detection with and without TRW (over {num_runs} runs)', fontsize=20)
plt.xlabel('Methods', fontsize=16)
plt.ylabel('Average Number of Nodes Detected', fontsize=16)
plt.xticks(rotation=45, fontsize=14)
plt.tight_layout()
plt.show()

# --- Print Average Metrics ---
for method in methods:
    print(f"\n{method}:")
    print(f"Precision: {np.mean(precision_runs[method]):.4f} ± {np.std(precision_runs[method]):.4f}")
    print(f"Recall:    {np.mean(recall_runs[method]):.4f} ± {np.std(recall_runs[method]):.4f}")
    print(f"F1-score:  {np.mean(f1_runs[method]):.4f} ± {np.std(f1_runs[method]):.4f}")

