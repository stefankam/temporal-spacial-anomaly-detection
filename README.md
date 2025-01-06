This folder contains:
a python file, ETH_data_collect_transactions_activity.py, to obtain and preprocess data according to 3.1 in the main text,
a python file, TRW_GCN_anomaly_detection.py, to create Figure 1 and 2, and Table 1.
a python file, TRW_GCN_anomaly_pattern.py, to create Figure 3.
a python file, TRW_GCN_complexity.py, to create Figure 4.
a python file, TRW_prob_sampling.py, to create Figure 5.

HOW do we obtain the accuracy of TGAT model ?
a python file, preprocess_ethereum_data.py, which should be added to TGAT repository downloaded from https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs by Xu et al. (ICLR 2020)tp preprocess the Ethereum data 
a python file, process.py, which needs to be replaced in this repository and run the command "python process.py" . Then run the command "python -u learn_edge.py -d eth_block_18168871_18168890_processed" as instructed in the repository. Finally, we could obtain the accuracy of the TGAT model. 