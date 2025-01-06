import networkx as nx
from web3 import Web3
from eth_utils import to_checksum_address
import statistics

# Connect to an Ethereum node using Web3.py
provider_url = 'https://eth-mainnet.g.alchemy.com/v2/WVW2asJD6jjPiuM50z073UCOd9jaJkVc'
web3 = Web3(Web3.HTTPProvider(provider_url))
latest_block = 9204477
#latest_block = web3.eth.block_number


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
end_block = latest_block - 1

# Create the transaction history graph for the specified block range
transaction_graph = create_transaction_history_graph(start_block, end_block)

# Print basic graph information
print("Transaction History Graph Information:")
print(transaction_graph)



import networkx as nx
from web3 import Web3
from eth_utils import to_checksum_address
import statistics

# Connect to an Ethereum node using Web3.py
provider_url = 'https://eth-mainnet.g.alchemy.com/v2/WVW2asJD6jjPiuM50z073UCOd9jaJkVc'
web3 = Web3(Web3.HTTPProvider(provider_url))
latest_block = 16839864

# latest_block = web3.eth.block_number


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
                               outgoing_value_list=[], incoming_tx_volume=0, outgoing_tx_volume=0,
                               incoming_value_variance=0, outgoing_value_variance=0, activity_rate=0,
                               change_in_activity=0, time_since_last=0, tx_volume=0,
                               ico_participation=False, flash_loan=False, token_airdrop=False, phishing=False,
                               frequent_large_transfers=False)
                graph.add_node(transaction['to'], outgoing_tx_count=0, incoming_tx_count=0, incoming_value_list=[],
                               outgoing_value_list=[], incoming_tx_volume=0, outgoing_tx_volume=0,
                               incoming_value_variance=0, outgoing_value_variance=0, activity_rate=0,
                               change_in_activity=0, time_since_last=0, tx_volume=0,
                               ico_participation=False, flash_loan=False, token_airdrop=False, phishing=False,
                               frequent_large_transfers=False)

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

        # Identify addresses with frequent and large transfers
        if (graph.nodes[node]['outgoing_tx_count'] >= 10) and (graph.nodes[node]['outgoing_tx_volume'] >= 100000):
            graph.nodes[node]['frequent_large_transfers'] = True

    return graph


# Set the block range for which you want to create the transaction history graph
start_block = latest_block
end_block = latest_block - 1

# Create the transaction history graph for the specified block range
transaction_graph = create_transaction_history_graph(start_block, end_block)

# Print basic graph information
print("Transaction History Graph Information:")
print(transaction_graph)








import networkx as nx
from web3 import Web3
from eth_utils import to_checksum_address
import statistics

# Connect to an Ethereum node using Web3.py
provider_url = 'https://eth-mainnet.g.alchemy.com/v2/WVW2asJD6jjPiuM50z073UCOd9jaJkVc'
web3 = Web3(Web3.HTTPProvider(provider_url))
#latest_block = web3.eth.block_number
latest_block = 18168890

# Function to calculate gas price
def calculate_gas_price(node_data):
    # Extract gas prices from outgoing transactions
    gas_prices = [tx['gas_price'] for tx in node_data.get('outgoing_transactions', [])]
    # Calculate average gas price or any other relevant metric
    avg_gas_price = sum(gas_prices) / len(gas_prices) if gas_prices else 0
    return avg_gas_price


# Function to detect token swaps
def detect_token_swaps(node_data):
    # Check if the node is involved in token swap transactions
    token_swaps = any(tx['type'] == 'token_swap' for tx in node_data.get('outgoing_transactions', []))
    return 1 if token_swaps else 0


# Function to check if a transaction is a token swap
def detect_smart_contract_interactions(node_data):
    # Check if the node interacts with smart contracts
    smart_contract_interactions = any(is_smart_contract(tx['to']) for tx in node_data.get('outgoing_transactions', []))
    return 1 if smart_contract_interactions else 0

def is_smart_contract(address):
    # Add your logic to determine if an address is a smart contract
    # For example, you can check if the address has bytecode associated with it
    # This is a simplified example, and you may need to refine it based on your requirements
    return True  # Placeholder, replace with actual implementation


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
                               outgoing_value_list=[], incoming_tx_volume=0, outgoing_tx_volume=0,
                               incoming_value_variance=0, outgoing_value_variance=0, activity_rate=0,
                               change_in_activity=0, time_since_last=0, tx_volume=0,
                               ico_participation=False, flash_loan=False, token_airdrop=False, phishing=False,
                               frequent_large_transfers=False, gas_price=0, token_swaps=0,
                               smart_contract_interactions=0)
                graph.add_node(transaction['to'], outgoing_tx_count=0, incoming_tx_count=0, incoming_value_list=[],
                               outgoing_value_list=[], incoming_tx_volume=0, outgoing_tx_volume=0,
                               incoming_value_variance=0, outgoing_value_variance=0, activity_rate=0,
                               change_in_activity=0, time_since_last=0, tx_volume=0,
                               ico_participation=False, flash_loan=False, token_airdrop=False, phishing=False,
                               frequent_large_transfers=False, gas_price=0, token_swaps=0,
                               smart_contract_interactions=0)

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

                # Add gas price, token swaps, and smart contract interactions
                graph.nodes[transaction['from']]['gas_price'] = calculate_gas_price(transaction)
                graph.nodes[transaction['to']]['gas_price'] = calculate_gas_price(transaction)
                graph.nodes[transaction['from']]['token_swaps'] = detect_token_swaps(transaction)
                graph.nodes[transaction['to']]['token_swaps'] = detect_token_swaps(transaction)
                graph.nodes[transaction['from']]['smart_contract_interactions'] = detect_smart_contract_interactions(transaction)
                graph.nodes[transaction['to']]['smart_contract_interactions'] = detect_smart_contract_interactions(transaction)

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

        # Identify addresses with frequent and large transfers
        if (graph.nodes[node]['outgoing_tx_count'] >= 10) and (graph.nodes[node]['outgoing_tx_volume'] >= 100000):
            graph.nodes[node]['frequent_large_transfers'] = True

    return graph


# Set the block range for which you want to create the transaction history graph
start_block = latest_block
end_block = latest_block - 20

# Create the transaction history graph for the specified block range
transaction_graph = create_transaction_history_graph(start_block, end_block)

# Print basic graph information
print("Transaction History Graph Information:")
print(transaction_graph)
