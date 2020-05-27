#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys,os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from GraphDQN import GraphDQN
import numpy as np
from tqdm import tqdm
import time
import networkx as nx
import pandas as pd
import pickle as cp
import graph


def GetSolution(STEPRATIO, MODEL_FILE_CKPT):
    ######################################################################################################################
    ##................................................Get Solution (model).....................................................
    dqn = GraphDQN()
    data_test_path = '../data/real/'
#     data_test_name = ['Crime','HI-II-14','Digg','Enron','Gnutella31','Epinions','Facebook','Youtube','Flickr']
    data_test_name = ['Crime','HI-II-14']
    model_file_path = './FINDER_CN/models/'
    model_file_ckpt = MODEL_FILE_CKPT
    model_file = model_file_path + model_file_ckpt
    ## save_dir
    save_dir = '../results/FINDER_CN/real'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    ## begin computing...
    print ('The best model is :%s'%(model_file))
    dqn.LoadModel(model_file)
    df = pd.DataFrame(np.arange(1*len(data_test_name)).reshape((1,len(data_test_name))),index=['time'], columns=data_test_name)
    #################################### modify to choose which stepRatio to get the solution
    stepRatio = STEPRATIO
    for j in range(len(data_test_name)):
        print ('\nTesting dataset %s'%data_test_name[j])
        data_test = data_test_path + data_test_name[j] + '.txt'
        solution, time = dqn.EvaluateRealData(model_file, data_test, save_dir, stepRatio)
        df.iloc[0,j] = time
        print('Data:%s, time:%.2f'%(data_test_name[j], time))
    save_dir_local = save_dir + '/StepRatio_%.4f' % stepRatio
    if not os.path.exists(save_dir_local):
        os.mkdir(save_dir_local)
    df.to_csv(save_dir_local + '/sol_time.csv', encoding='utf-8', index=False)


def EvaluateSolution(STEPRATIO, MODEL_FILE_CKPT, STRTEGYID):
    #######################################################################################################################
    ##................................................Evaluate Solution.....................................................
    dqn = GraphDQN()
    data_test_path = '../data/real/'
#     data_test_name = ['Crime', 'HI-II-14', 'Digg', 'Enron', 'Gnutella31', 'Epinions', 'Facebook', 'Youtube', 'Flickr']
    data_test_name = ['Crime','HI-II-14']
    save_dir = '../results/FINDER_CN/real/StepRatio_%.4f/'%STEPRATIO
    ## begin computing...
    df = pd.DataFrame(np.arange(2 * len(data_test_name)).reshape((2, len(data_test_name))), index=['solution', 'time'], columns=data_test_name)
    for i in range(len(data_test_name)):
        print('\nEvaluating dataset %s' % data_test_name[i])
        data_test = data_test_path + data_test_name[i] + '.txt'
        solution = save_dir + data_test_name[i] + '.txt'
        t1 = time.time()
        # strategyID: 0:no insert; 1:count; 2:rank; 3:multiply
        ################################## modify to choose which strategy to evaluate
        strategyID = STRTEGYID
        score, MaxCCList = dqn.EvaluateSol(data_test, solution, strategyID, reInsertStep=0.001)
        t2 = time.time()
        df.iloc[0, i] = score
        df.iloc[1, i] = t2 - t1
        result_file = save_dir + '/MaxCCList_Strategy_' + data_test_name[i] + '.txt'
        with open(result_file, 'w') as f_out:
            for j in range(len(MaxCCList)):
                f_out.write('%.8f\n' % MaxCCList[j])
        print('Data: %s, score:%.6f' % (data_test_name[i], score))
    df.to_csv(save_dir + '/solution_score.csv', encoding='utf-8', index=False)


def main():
    model_file_ckpt = 'nrange_30_50_iter_93300.ckpt'
    GetSolution(0.01, model_file_ckpt)
    EvaluateSolution(0.01, model_file_ckpt, 0)


if __name__=="__main__":
    main()
