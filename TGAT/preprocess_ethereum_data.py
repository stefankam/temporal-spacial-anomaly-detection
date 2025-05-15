import csv
import os
import pickle
import pandas as pd
import ast
import numpy as np

# Load the graph
graph = pickle.load(open('data/eth_latest_100_block.pickle', 'rb'))

# Folder to save the processed data
output_folder = 'ETH_processed'
os.makedirs(output_folder, exist_ok=True)

# File to save the CSV
output_file = os.path.join(output_folder, 'eth_latest_100_block_processed.csv')

# Headers for node features
node_feature_headers = [
    'outgoing_tx_count', 'incoming_tx_count', 'incoming_value_list', 'outgoing_value_list',
    'incoming_tx_volume', 'outgoing_tx_volume', 'incoming_value_variance', 'outgoing_value_variance',
    'activity_rate', 'change_in_activity', 'time_since_last', 'tx_volume', 'ico_participation',
    'flash_loan', 'token_airdrop', 'phishing', 'frequent_large_transfers', 'gas_price',
    'token_swaps', 'smart_contract_interactions', 'last_transaction_block'
]

# Function to extract node feature values as a comma-separated string
def get_node_features(graph, node, feature_headers):
    features = graph.nodes[node]  # Get features for the node
    feature_values = [str(features.get(header, "")) for header in feature_headers]  # Extract values for each feature
    return ",".join(feature_values)  # Return as a comma-separated string

# Open the CSV file for writing
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)

    # Write the header row
    header = ['user', 'item', 'timestamp', 'state_label'] + [f'source_{header}' for header in node_feature_headers] + [f'target_{header}' for header in node_feature_headers]
    writer.writerow(header)

    # Iterate over all edges and write the data to the CSV
    for edge in graph.edges(data=True):
        source, target, edge_attrs = edge  # Unpack the source, target, and attributes

        # Extract the edge attributes: timestamp is required, others are optional
        timestamp = edge_attrs.get('timestamp', "")
#        state_label = edge_attrs.get('weight', "")  # Assuming 'weight' could be the state_label
        state_label = int(graph.nodes[source].get('change_in_activity', 0))

        # Get the node feature values as comma-separated strings
        source_features = get_node_features(graph, source, node_feature_headers)
        target_features = get_node_features(graph, target, node_feature_headers)

        # Write the row to the CSV
        writer.writerow([source, target, timestamp, state_label] + source_features.split(",") + target_features.split(","))

print(f"CSV file saved at {output_file}")

"CLEANING all String items"
# Step 1: Load the CSV file into a pandas DataFrame
df = pd.read_csv(f'{output_file}')

# Step 2: Identify columns where all values are empty lists `[]`
cols_to_drop = [col for col in df.columns if df[col].apply(lambda x: x == '[]').all()]

# Step 3: Drop the identified columns
df_cleaned = df.drop(cols_to_drop, axis=1)

# Step 4: Remove columns that contain string values "True" or "False"
cols_to_remove = [col for col in df_cleaned.columns if df_cleaned[col].apply(lambda x: str(x) in ["True", "False"]).any()]
df_cleaned = df_cleaned.drop(cols_to_remove, axis=1)

# Step 5: Remove brackets and convert string representations of lists to their values
def remove_brackets(x):
    if isinstance(x, str):
        try:
            # Convert string representation of list to actual list
            value = ast.literal_eval(x)
            if isinstance(value, list):
                return value[0] if value else np.nan  # Return first element if it's a non-empty list
        except (ValueError, SyntaxError):
            return x.strip('[]')  # Remove brackets if not a valid list
    return x  # Return the original value if it's not a string

df_cleaned = df_cleaned.applymap(remove_brackets)

# Step 6: Save the cleaned DataFrame back to a CSV (optional)
df_cleaned.to_csv(f'{output_file}', index=False)