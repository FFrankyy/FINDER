#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys,os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from FINDER import FINDER
from tqdm import tqdm


def main():
    dqn = FINDER()
    data_test_path = '../data/synthetic/uniform_cost/'
#     data_test_name = ['30-50', '50-100', '100-200', '200-300', '300-400', '400-500']
    data_test_name = ['30-50', '50-100']
    model_file = './FINDER_ND/models/nrange_30_50_iter_78000.ckpt'
    
    file_path = '../results/FINDER_ND/synthetic'
    
    if not os.path.exists('../results/FINDER_ND'):
        os.mkdir('../results/FINDER_ND')
    if not os.path.exists('../results/FINDER_ND/synthetic'):
        os.mkdir('../results/FINDER_ND/synthetic')
        
    with open('%s/result.txt'%file_path, 'w') as fout:
        for i in tqdm(range(len(data_test_name))):
            data_test = data_test_path + data_test_name[i]
            score_mean, score_std, time_mean, time_std = dqn.Evaluate(data_test, model_file)
            fout.write('%.2f±%.2f,' % (score_mean * 100, score_std * 100))
            fout.flush()
            print('\ndata_test_%s has been tested!' % data_test_name[i])


if __name__=="__main__":
    main()
