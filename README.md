# alphamev-submission
Submission code for the first https://alphamev.ai MEV competition with AUC (0.9812) and MSE (0.122).

Competition was to build a model to predict whether Ethereum transactions were backrunnable and how much MEV profit could be extracted from the transaction.

Features were generated from Ethereum transaction metadata such as number of each call type and size of transaction data.  In addition, a hardcoded list of popular DeFi smart contract addresses was used to identify which transactions were being sent to a DeFi protocol.

Classification and Regression used AutoML from PyCaret (https://pycaret.org/).


