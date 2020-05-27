import networkx as nx
import numpy as np
import random



data_path = 'E:/Ubuntu_home/Code/My Code/GRAPHDQN/Server version/Data&Solution/Real_Data/Data/'
data_name = ['maayan-figeys','maayan-Stelzl','maayan-vidal','maayan-yeast']
save_dir = 'E:/Ubuntu_home/Code/My Code/GRAPHDQN/Server version/Data&Solution/Real_Data/Data_Cost'

cost_type = 'degree_noise'

for i in range(len(data_name)):
    data = data_path + data_name[i] + '.txt'
    g = nx.read_edgelist(data)

    ### degree weight
    if cost_type == 'degree':
        degree = nx.degree(g)
        maxDegree = max(dict(degree).values())
        weights = {}
        for node in g.nodes():
            weights[node] = degree[node] / maxDegree

    elif cost_type == 'degree_noise':
        degree = nx.degree(g)
        median_val = np.median(list(dict(degree).values()))
        weights = {}
        for node in g.nodes():
            episilon = 0.5 * median_val * np.random.normal(0, 1)
            weights[node] = 0.5 * degree[node] + episilon
            if weights[node] < 0.0:
                weights[node] = - weights[node]
        max_weight = np.max(list(weights.values()))
        for node in g.nodes():
            weights[node] = weights[node] / max_weight

    elif cost_type == 'random':
    # ### random weight
        weights = {}
        for node in g.nodes():
            weights[node] = random.uniform(0, 1)

    nx.set_node_attributes(g, weights, 'weight')
    save_dir_g = '%s/%s_%s.gml' % (save_dir, data_name[i], cost_type)
    nx.write_gml(g, save_dir_g)


    f_weight = open('%s/%s_%s_weight.txt'%(save_dir, data_name[i], cost_type), 'w')
    for j in range(nx.number_of_nodes(g)):
        f_weight.write('%.8f\n'%weights[str(j)])
        f_weight.flush()
    f_weight.close()

    print ('Data %s is finished!'%data_name[i])
