This folder contains:
a python file, ETH_data_collect_transactions_activity.py, to obtain and preprocess data according to 3.1 in the main text,
a python file, TRW_GCN_anomaly_detection.py, to create Figure 1 and 2, and Table 2.
a python file, TRW_GCN_anomaly_pattern.py, to create Figure 3.
a python file, TGAT_anomaly_pattern.py, with the TGAT folder to create Table 4.
a python file, TRW_prob_sampling.py, to create Figure 4.

HOW do we implement TGAT model for our blockchain data ?
1. we generated a python file, preprocess_ethereum_data.py, which should be added to TGAT repository downloaded from https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs by Xu et al. (ICLR 2020) to preprocess the Ethereum data 
2. a python file, ETH_process.py, which needs to be also added to this repository and run the command "python ETH_process.py". This creates thre files in the folder ETH_processed: ml_eth_latest_100_block_processed.npy ml_eth_latest_100_block_processed.csv  ml_eth_latest_100_block_processed_node.npy
3. run the command "python TGAT_anomaly_detection.py -d eth_block_100_latest_processed" as instructed in the repository. You should be able to obtain the results of Table 4. 