import pickle as cp
import networkx as nx
import os
from tqdm import tqdm
import random

data_test_path = 'E:/Ubuntu_home/Code/GRAPHDQN/Server version/Data&Solution/Synthesized_Data/gtype-barabasi_albert-nrange-'
data_test_name = ['15-20','30-50','50-100','100-200','200-300','300-400','400-500','500-600','1000-1200']
data_test_suffix = '-n_graph-1000-p-0.00-m-4.pkl'

save_dir_1 = 'E:/Ubuntu_home/Code/GRAPHDQN/Server version/Data&Solution/Synthesized_Data/gml_randomCost/'
n_test = 100
for i in range(len(data_test_name)):
    print ('\ndata_test%d'%i)
    data_test = data_test_path + data_test_name[i] + data_test_suffix
    save_dir_2 = data_test_name[i]
    save_dir = save_dir_1 + save_dir_2
    if not os.path.exists(save_dir):    #make dir
        os.mkdir(save_dir)
    f = open(data_test, 'rb')
    for j in tqdm(range(n_test)):
        g = cp.load(f)

        ### random weight
        weights = {}
        for node in g.nodes():
            weights[node] = random.uniform(0, 1)

        # ### degree weight
        # degree = nx.degree(g)
        # maxDegree = max(degree.values())
        # weights = {}
        # for node in g.nodes():
        #     weights[node] = degree[node] / maxDegree

        nx.set_node_attributes(g, 'weight', weights)
        save_dir_g = '%s/g_%d'%(save_dir,j)
        nx.write_gml(g, save_dir_g)


# save_dir_11 = 'E:/Ubuntu_home/Code/GRAPHDQN/Server version/Data&Solution/Synthesized_Data/gml_randomCost/'
# n_test = 100
# for i in range(len(data_test_name)):
#     print ('data_test%d'%i)
#     data_test = data_test_path + data_test_name[i] + data_test_suffix
#     save_dir_22 = data_test_name[i]
#     save_dir = save_dir_11 + save_dir_22
#     if not os.path.exists(save_dir):    #make dir
#         os.mkdir(save_dir)
#     f = open(data_test, 'rb')
#     for j in tqdm(range(n_test)):
#         g = cp.load(f)
#         save_dir_g = '%s/g_%d'%(save_dir,j)
#         nx.write_gml(g, save_dir_g)