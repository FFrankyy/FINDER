#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys,os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from GraphDQN import GraphDQN
from tqdm import tqdm

def main():
    dqn = GraphDQN()
    data_test_path = '../data/synthetic/uniform_cost/'
#     data_test_name = ['30-50', '50-100', '100-200', '200-300', '300-400', '400-500']
    data_test_name =['30-50', '50-100']
    model_file = './FINDER_CN/models/nrange_30_50_iter_93300.ckpt'
    
    file_path = '../results/FINDER_CN/synthetic'
#     if not os.path.exists(file_path):
    if not os.path.exists('../results/FINDER_CN'):
        os.mkdir('../results/FINDER_CN')
    if not os.path.exists('../results/FINDER_CN/synthetic'):
        os.mkdir('../results/FINDER_CN/synthetic')
        
    with open('%s/result.txt'%file_path, 'w') as fout:
        for i in tqdm(range(len(data_test_name))):
            data_test = data_test_path + data_test_name[i]
            score_mean, score_std, time_mean, time_std = dqn.Evaluate(data_test, model_file)
            fout.write('%.2f±%.2f,' % (score_mean * 100, score_std * 100))
            fout.flush()
            print('data_test_%s has been tested!' % data_test_name[i])


if __name__=="__main__":
    main()
