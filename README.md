# Bitcoin-Mining-Pool-Detection-and-Identifying-Anomalies
Bitcoin provides an extra level of anonymity for the identities of the users, since the user addresses are hashed. A Mining pool is the pooling of resources by miners, who share their processing power over a network, to split the reward equally, according to the amount of work they contributed to the probability of finding a block. I attempt to create a classifier model which identifies the mining pools from other users and then detects anomalies in these mining pools.

Please run the Script.py file.

Script.py file takes "miningdata.csv" as input and performs analysis on the data. "miningdata.csv" is bitcoin historic data made available online by Google Cloud Platform through Big Query API.

I used the following SQL query to select a few numeric attributes with  miner/non-miner labels and addresses and then downloaded the data in csv format.(https://gist.github.com/allenday/16cf63fb6b3ed59b78903b2d414fe75b) Please select the attributes as required.

Read "Report.pdf" for a detailed explanation of the code and the results.

The code returns two image files after running "feature_imp.png" and "anomaly.png". They are shown in the report as well.
