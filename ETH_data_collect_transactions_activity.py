import networkx as nx
from web3 import Web3
from eth_utils import to_checksum_address
import statistics

# Connect to an Ethereum node using Web3.py
provider_url = 'https://eth-mainnet.g.alchemy.com/v2/WVW2asJD6jjPiuM50z073UCOd9jaJkVc'
web3 = Web3(Web3.HTTPProvider(provider_url))
latest_block = web3.eth.block_number


def create_transaction_history_graph(start_block, end_block):
    # Initialize an empty graph
    graph = nx.DiGraph()
    counter = 0
    duration = start_block - end_block + 1
    prev_tx_counts = {}

    # Iterate over the blocks in reverse order
    for block_number in range(start_block, end_block, -1):
        block = web3.eth.get_block(block_number)
        print(block_number)
        counter += 1
        print(counter)

        # Iterate over the transactions in the block
        try:
            for tx_hash in block['transactions']:
                transaction = web3.eth.get_transaction(tx_hash.hex())

                # Initialize attributes for nodes
                graph.add_node(transaction['from'], outgoing_tx_count=0, incoming_tx_count=0, incoming_value_list=[],
                               outgoing_value_list=[], incoming_tx_volume=0, outgoing_tx_volume=0)
                graph.add_node(transaction['to'], outgoing_tx_count=0, incoming_tx_count=0, incoming_value_list=[],
                               outgoing_value_list=[], incoming_tx_volume=0, outgoing_tx_volume=0)

                # Count the transactions and sum the values for volumes
                graph.nodes[transaction['from']]['outgoing_tx_count'] += 1
                graph.nodes[transaction['to']]['incoming_tx_count'] += 1
                graph.nodes[transaction['from']]['outgoing_tx_volume'] += int(transaction['value'])
                graph.nodes[transaction['to']]['incoming_tx_volume'] += int(transaction['value'])

                # Track last transaction block
                graph.nodes[transaction['from']]['last_transaction_block'] = block_number
                graph.nodes[transaction['to']]['last_transaction_block'] = block_number

                # Store the value of each transaction in lists (useful for variance calculations)
                graph.nodes[transaction['from']]['outgoing_value_list'].append(int(transaction['value']))
                graph.nodes[transaction['to']]['incoming_value_list'].append(int(transaction['value']))

                # Add the transaction as an edge with weight and timestamp attributes
                graph.add_edge(transaction['from'], transaction['to'],
                               hash=transaction['hash'].hex(),
                               weight=int(transaction['value']),
                               timestamp=block['timestamp'])
        except:
            pass

    # Post-processing nodes
    for node in graph.nodes():
        # Variance calculations for transaction values
        incoming_values = graph.nodes[node].get('incoming_value_list', [])
        outgoing_values = graph.nodes[node].get('outgoing_value_list', [])
        graph.nodes[node]['incoming_value_variance'] = statistics.variance(incoming_values) if len(
            incoming_values) >= 2 else 0
        graph.nodes[node]['outgoing_value_variance'] = statistics.variance(outgoing_values) if len(
            outgoing_values) >= 2 else 0

        # Calculate activity metrics
        total_tx_count = graph.nodes[node].get('incoming_tx_count', 0) + graph.nodes[node].get('outgoing_tx_count', 0)
        graph.nodes[node]['activity_rate'] = total_tx_count / duration
        prev_tx_count = prev_tx_counts.get(node, 0)
        graph.nodes[node]['change_in_activity'] = total_tx_count - prev_tx_count
        prev_tx_counts[node] = total_tx_count
        graph.nodes[node]['time_since_last'] = start_block - graph.nodes[node].get('last_transaction_block',
                                                                                   start_block)

        # Calculate total transaction volume
        graph.nodes[node]['tx_volume'] = graph.nodes[node]['incoming_tx_volume'] + graph.nodes[node][
            'outgoing_tx_volume']

    return graph


# Set the block range for which you want to create the transaction history graph
start_block = latest_block
end_block = latest_block - 2

# Create the transaction history graph for the specified block range
transaction_graph = create_transaction_history_graph(start_block, end_block)

# Print basic graph information
print("Transaction History Graph Information:")
print(transaction_graph)
