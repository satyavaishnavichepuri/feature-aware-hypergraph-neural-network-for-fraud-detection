[exact code redacted to protect research sensitivity until official teams are formed. basic overview attached]

This project focuses on detecting fraudulent transactions by modeling group-level behavior instead of analyzing transactions individually.

The pipeline groups temporally aligned transactions using clustering techniques and represents them as hypergraphs, where each hyperedge captures relationships between multiple transactions. These hyperedges are evaluated using similarity and compactness measures, and low-quality connections are removed to retain only meaningful structures.

The refined graph is then converted into a bipartite graph, and a Graph Convolutional Network (GCN) is trained to learn patterns across transaction groups. The model is evaluated using temporal splits to simulate real-world scenarios and improve robustness, scalability, and reliability in fraud detection.
