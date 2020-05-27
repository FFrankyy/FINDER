#ifndef NSTEP_REPLAY_MEM_H
#define NSTEP_REPLAY_MEM_H

#include <vector>
#include <random>
#include "graph.h"
#include "mvc_env.h"


class ReplaySample
{
public:
    ReplaySample(int batch_size);
    std::vector< std::shared_ptr<Graph> > g_list;
    std::vector< std::vector<int>> list_st, list_s_primes;
    std::vector<int> list_at;
    std::vector<double> list_rt;
    std::vector<bool> list_term;
};

class NStepReplayMem
{
public:
      NStepReplayMem(int memory_size);

     void Add(std::shared_ptr<Graph> g,
                    std::vector<int> s_t,
                    int a_t, 
                    double r_t,
                    std::vector<int> s_prime,
                    bool terminal);

     void Add(std::shared_ptr<MvcEnv> env,int n_step);

     std::shared_ptr<ReplaySample> Sampling(int batch_size);
     std::vector< std::shared_ptr<Graph> > graphs;
     std::vector<int> actions;
     std::vector<double> rewards;
     std::vector< std::vector<int> > states, s_primes;
     std::vector<bool> terminals;

     int current, count, memory_size;
     std::default_random_engine generator;
     std::uniform_int_distribution<int>* distribution;
};

#endif