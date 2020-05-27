import networkx as nx
import numpy as np
import random



data_path = './Data/InternetP2P/p2p-Gnutella'
data_name = ['04','05','06','08','09','24','25','30','31']
save_dir = './Data/InternetP2P_cost/'
costType = {0: 'degree', 1:'random'}

for k in range(2):
    for i in range(len(data_name)):
        data = data_path + data_name[i] + '.txt'
        g = nx.read_weighted_edgelist(data)

        if k == 0:  ### degree weight
            degree = nx.degree(g)
            maxDegree = max(degree.values())
            weights = {}
            for node in g.nodes():
                weights[node] = degree[node] / maxDegree

        else:  ### random weight
            weights = {}
            for node in g.nodes():
                weights[node] = random.uniform(0, 1)

        nx.set_node_attributes(g, 'weight', weights)
        save_dir_g = '%s/p2p-Gnutella%s_%s.gml' % (save_dir, data_name[i], costType[k])
        nx.write_gml(g, save_dir_g)
