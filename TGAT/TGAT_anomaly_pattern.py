import random
import matplotlib.pyplot as plt
import pickle

import torch_geometric
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
import numpy as np
import pickle
import random
import matplotlib.pyplot as plt
import torch_geometric.utils
import torch
import networkx as nx
import pickle
from torch_geometric.data import Data
from module import TGAN
from graph import NeighborFinder
import pandas as pd
from collections import defaultdict
import torch.nn.functional as F

# Load the data
df = pd.read_csv('./ETH_processed/ml_eth_latest_100_block_processed.csv')
e_feat = np.load('./ETH_processed/ml_eth_latest_100_block_processed.npy')
n_feat = np.load('./ETH_processed/ml_eth_latest_100_block_processed_node.npy')

# Prepare edge index, timestamps, etc.
# df =  df[:10]
source_nodes = df['u'].values
destination_nodes = df['i'].values
edge_idxs = df['idx'].values
timestamps = df['ts'].values

df = df.sort_values(by='ts')  # sort by timestamp if not already# Build adjacency list for ngh_finder
df = df.dropna(subset=['u', 'i', 'ts', 'label'])
n_feat = np.nan_to_num(n_feat)
e_feat = np.nan_to_num(e_feat)

ts_min = min(df['ts'])
node_freqs = defaultdict(int)
adj_list = [[] for _ in range(n_feat.shape[0])]
for src, dst, eid, ts in zip(source_nodes, destination_nodes, edge_idxs, timestamps):
    adj_list[src].append((dst, eid, (ts - ts_min)))
    adj_list[dst].append((src, eid, (ts - ts_min)))
    node_freqs[int(src)] += 1
    node_freqs[int(dst)] += 1

# Initialize NeighborFinder
full_ngh_finder = NeighborFinder(adj_list, uniform=False)  # or True if sampling uniformly
print(full_ngh_finder.__dict__)

# Split data
train_ratio = 0.8
num_interactions = len(df)
train_size = int(train_ratio * num_interactions)
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]
train_indices = train_df.index.values
test_indices = test_df.index.values
print("train_indices shape: ", train_indices.shape)
print("test_indices shape", test_indices.shape)

# Set device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate TGAN
model = TGAN(
    ngh_finder=full_ngh_finder,
    n_feat=n_feat,
    e_feat=e_feat,
    attn_mode='prod',
    use_time='time',
    agg_method='attn',
    num_layers=2,
    n_head=2,
    drop_out=0.1,
)

# Define loss and optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Training loop
model.train()
batch_size = 32  # or whatever fits in memory
for epoch in range(1):
    for start in range(0, len(train_indices), batch_size):
        end = start + batch_size
        batch_idx = train_indices[start:end]
        src_np = np.array(df.loc[batch_idx, 'u'].tolist())
        dst_np = np.array(df.loc[batch_idx, 'i'].tolist())
        ts_np = np.array(df.loc[batch_idx, 'ts'].tolist())
        label = np.array(df.loc[batch_idx, 'label'].astype(float).tolist())

        # Normalize timestamp
        ts_np = ts_np - ts_np.min()

        optimizer.zero_grad()
        pred = model(src_np, dst_np, ts_np)

        embedding_dim = pred.shape
        print(f"train index {start}: Label = {label}, Pred = {pred}")

        loss = criterion(pred, torch.tensor(label, dtype=torch.float))
        print("loss: ", loss)
        loss.backward()
        optimizer.step()

# train_val(train_df, model, optimizer, criterion, device=device, batch_size=2, n_epoch=1)


# --- 1. Run TGAN on all nodes ---
model.eval()
all_embeddings = []

batch_size = 32
with torch.no_grad():  # Disable gradient calculation
    for start in range(0, len(train_indices), batch_size):
        end = start + batch_size
        batch_idx = train_indices[start:end]

        # Convert batch_idx to numpy arrays
        src_np = np.array(df.loc[batch_idx, 'u'].tolist())
        dst_np = np.array(df.loc[batch_idx, 'i'].tolist())
        ts_np = np.array(df.loc[batch_idx, 'ts'].tolist())

        # Ensure timestamps are normalized if necessary
        ts_np = ts_np - ts_min

        # Forward pass to get predictions (or embeddings)
        emb = model(src_np, dst_np, ts_np)

        # Collect the embeddings
        all_embeddings.append(emb.cpu())

# with torch.no_grad():
#    for target_node in range(n_feat.shape[0]):
#        src = np.array([[target_node]], dtype=np.int64)
#        dst = np.array([[target_node]], dtype=np.int64)
#        ts = np.array([[latest_timestamp]], dtype=np.float32)

#        try:
#           emb = model(src, dst, ts)
#           print(f"Embedding shape before squeeze: {emb.shape}")
#           emb = emb.squeeze(0).cpu()
#        except Exception as e:
#           print(f"Node {target_node} failed: {e}")
#           emb = torch.zeros(embedding_dim)
#           print("emb: ", emb)
#           all_embeddings.append(emb)


# Determine the max embedding length
max_len = max(emb.shape[0] for emb in all_embeddings)

# Pad embeddings to max_len
padded_embeddings = [F.pad(emb, (0, max_len - emb.shape[0])) for emb in all_embeddings]

# Now safe to stack
embeddings = torch.stack(padded_embeddings)
print("embeddings:", embeddings)
print("All embeddings shape:", embeddings.shape)

# --- 2. Split into train/test ---
num_nodes = embeddings.shape[0]
indices = np.arange(num_nodes)
np.random.shuffle(indices)
split = int(0.5 * num_nodes)
train_indices = indices[:split]
test_indices = indices[split - 1:]

train_embeddings = embeddings[train_indices]
test_embeddings = embeddings[test_indices]

# --- 3. Prepare node frequencies ---
test_node_freq_values = [node_freqs[int(idx)] for idx in indices if int(idx) in node_freqs]
print("test_node_freq_values : ", test_node_freq_values)


# --- 4. Compute anomaly scores ---
def compute_anomaly_scores(embeddings, node_freqs_values):
    scores = []
    for idx, embedding in enumerate(embeddings):
        mean = embedding.mean().item()
        std = embedding.std().item() + 1e-6
        latest_value = embedding[-1].item()
        print("latest_value: ", latest_value)
        z_score = (latest_value - mean) / std
        print("node_freqs.get(int(idx), 0): ", node_freqs.get(int(idx), 0))
        weighted_z_score = z_score * node_freqs.get(int(idx), 0)
        print("weighted_z_score: ", weighted_z_score)
        scores.append(weighted_z_score)
    return scores


test_anomaly_scores = compute_anomaly_scores(embeddings, test_node_freq_values)
print("test_anomaly_scores: ", test_anomaly_scores)

# --- 5. Define true labels ---
test_true_labels = []
for idx, emb in zip(indices, embeddings):
    test_true_labels.append(df.loc[idx, 'label'])

print("Scores:", len(test_anomaly_scores))
print("Labels:", len(test_true_labels))


# --- 6. Evaluate using thresholds ---
def evaluate_anomalies(anomaly_scores, true_labels, thresholds):
    precisions, recalls, f1_scores, anomaly_counts = [], [], [], []
    for threshold in thresholds:
        detected_labels = [1 if abs(score) > threshold else 0 for score in anomaly_scores]
        precisions.append(precision_score(true_labels, detected_labels, zero_division=0))
        recalls.append(recall_score(true_labels, detected_labels, zero_division=0))
        f1_scores.append(f1_score(true_labels, detected_labels, zero_division=0))
        # Count anomalies: sum of detected anomalies (1's)
        anomaly_count = sum(detected_labels)
        anomaly_counts.append(anomaly_count)
    return np.mean(precisions), np.mean(recalls), np.mean(f1_scores), anomaly_counts


thresholds = [0.5, 1.0, 1.5, 2.0]
avg_precision, avg_recall, avg_f1_score, anomaly_counts = evaluate_anomalies(test_anomaly_scores, test_true_labels,
                                                                             thresholds)

# --- 7. Print Results ---
print(f"Average Precision: {avg_precision:.3f}")
print(f"Average Recall: {avg_recall:.3f}")
print(f"Average F1-score: {avg_f1_score:.3f}")

# Print number of anomalies detected for each threshold
for threshold, anomaly_count in zip(thresholds, anomaly_counts):
    print(f"Threshold: {threshold}, Detected Anomalies: {anomaly_count}")