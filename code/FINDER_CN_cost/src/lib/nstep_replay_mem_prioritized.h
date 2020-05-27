#ifndef NSTEP_REPLAY_MEM_PRIORITIZED_H
#define NSTEP_REPLAY_MEM_PRIORITIZED_H

#include <vector>
#include <random>
#include "graph.h"
#include "mvc_env.h"


class Data{
public:
    Data();
    std::shared_ptr<Graph> g;
    std::vector<int> s_t;
    std::vector<int> s_prime;
    int a_t;
    double r_t;
    bool term_t;
};

class LeafResult{
public:
    LeafResult();
    int leaf_idx;
    double p;
    std::shared_ptr<Data> data;
};

class SumTree
{
public:
    SumTree(int capacity);
    int capacity;
    int data_pointer;
    //bool isOverWrite;
    double minElement;
    double maxElement;
    std::vector<double> tree;
    std::vector<std::shared_ptr<Data>> data;
    void Add(double p, std::shared_ptr<Data> data);
    void Update(int tree_idx,double p);
    std::shared_ptr<LeafResult> Get_leaf(double v);
};


class ReplaySample{
public:
    ReplaySample(int batch_size);
    std::vector<int> b_idx;
    std::vector<double> ISWeights;
    std::vector< std::shared_ptr<Graph> > g_list;
    std::vector< std::vector<int>> list_st, list_s_primes;
    std::vector<int> list_at;
    std::vector<double> list_rt;
    std::vector<bool> list_term;
};


class Memory{
public:
    Memory(double epsilon,double alpha,double beta,double beta_increment_per_sampling,double abs_err_upper,int capacity);
    std::shared_ptr<SumTree> tree;
    double epsilon;
    double alpha;
    double beta;
    double beta_increment_per_sampling;
    double abs_err_upper;
    void Store(std::shared_ptr<Data> transition);
    void Add(std::shared_ptr<MvcEnv> env,int n_step);
    std::shared_ptr<ReplaySample> Sampling(int n);
    void batch_update(std::vector<int> tree_idx, std::vector<double> abs_errors);
};

#endif