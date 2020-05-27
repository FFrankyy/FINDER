#include "nstep_replay_mem_prioritized.h"
#include "i_env.h"
#include <cassert>
#include <algorithm>
#include <time.h>
#include <math.h>

#define max(x, y) (x > y ? x : y)
#define min(x, y) (x > y ? y : x)


Data::Data()
{

}

LeafResult::LeafResult()
{

}
SumTree::SumTree(int _capacity){
    capacity = _capacity;
    tree.resize(2 * capacity - 1);
    data.resize(capacity);
    data_pointer = 0;
    minElement = 2147483647/2;
    maxElement = 0.0;
    //isOverWrite = false;
}
void SumTree::Add(double p,std::shared_ptr<Data> _data){
    int tree_idx = data_pointer + capacity - 1;
    data[data_pointer] = _data;
    Update(tree_idx, p);
    data_pointer += 1;
    if (data_pointer >= capacity)
    {
        //isOverWrite = true;
        data_pointer = 0;
    }
}

void SumTree::Update(int tree_idx,double p){
    double change = p - tree[tree_idx];
    tree[tree_idx] = p;
    assert(tree_idx>=capacity-1&&tree_idx<2*capacity-1);
    assert(p > 0);
    if(p < minElement)
    {
        minElement = p;
    }

    if(p > maxElement)
    {
        maxElement = p;
    }

    while(tree_idx != 0){
         tree_idx = (int)((tree_idx - 1)/2);
         tree[tree_idx] += change;
    }
}

std::shared_ptr<LeafResult> SumTree::Get_leaf(double v){
    int parent_idx = 0;
    int leaf_idx;
    while(true){
        int cl_idx = 2 * parent_idx + 1;
        int cr_idx = cl_idx + 1;
        if(cl_idx >= (int)tree.size()){
            leaf_idx = parent_idx;
            break;
        }
        else{
            if(v <= tree[cl_idx]){
                parent_idx = cl_idx;
            }
            else{
                v -= tree[cl_idx];
                parent_idx = cr_idx;
            }
        }
    }
    int data_idx = leaf_idx - capacity + 1;
    std::shared_ptr<LeafResult> result =std::shared_ptr<LeafResult>(new LeafResult());
    result->leaf_idx = leaf_idx;
    result->p = tree[leaf_idx];
    result->data = data[data_idx];
    return result;
}

ReplaySample::ReplaySample(int batch_size){
    b_idx.resize(batch_size);
    ISWeights.resize(batch_size);
    g_list.resize(batch_size);
    list_st.resize(batch_size);
    list_s_primes.resize(batch_size);
    list_at.resize(batch_size);
    list_rt.resize(batch_size);
    list_term.resize(batch_size);
}

Memory::Memory(double _epsilon,double _alpha,double _beta,double _beta_increment_per_sampling,double _abs_err_upper,int capacity){
    tree =std::shared_ptr<SumTree>(new SumTree(capacity));
    epsilon = _epsilon;
    alpha = _alpha;
    beta = _beta;
    beta_increment_per_sampling = _beta_increment_per_sampling;
    abs_err_upper = _abs_err_upper;
}


void Memory::Store(std::shared_ptr<Data> transition){
//    double max_p = *max_element(tree->tree.end()-tree->capacity,tree->tree.end());
    double max_p = tree->maxElement;
    if(max_p == 0){
        max_p = abs_err_upper;
    }
    tree->Add(max_p, transition);
}

void Memory::Add(std::shared_ptr<MvcEnv> env,int n_step)
{
    assert(env->isTerminal());
    int num_steps = (int)env->state_seq.size();
    assert(num_steps);

    env->sum_rewards[num_steps - 1] = env->reward_seq[num_steps - 1];
    for (int i = num_steps - 1; i >= 0; --i)
        if (i < num_steps - 1)
            env->sum_rewards[i] = env->sum_rewards[i + 1] + env->reward_seq[i];

    for (int i = 0; i < num_steps; ++i)
    {
        bool term_t = false;
        double cur_r;
        std::vector<int> s_prime;
        if (i + n_step >= num_steps)
        {
            cur_r = env->sum_rewards[i];
            s_prime = (env->action_list);
            term_t = true;
        } else {
            cur_r = env->sum_rewards[i] - env->sum_rewards[i + n_step];
            s_prime = (env->state_seq[i + n_step]);
        }
        std::shared_ptr<Data> transition =std::shared_ptr<Data>(new Data());
        transition->g = env->graph;
        transition->s_t = env->state_seq[i];
        transition->s_prime = s_prime;
        transition->a_t = env->act_seq[i];
        transition->r_t = cur_r;
        transition->term_t = term_t;
        Store(transition);
    }
}



std::shared_ptr<ReplaySample> Memory::Sampling(int n){
    std::shared_ptr<ReplaySample> result =std::shared_ptr<ReplaySample>(new ReplaySample(n));
    double total_p = tree->tree[0];
    double pri_seg = total_p / n;
    beta = min(1.0, beta + beta_increment_per_sampling);

    double min_prob = tree->minElement/total_p;
//    printf ("min_prob:%.6f\n",min_prob);

//    double sum=0.0;
    for(int i = 0;i < n;++i){
        double a = pri_seg * i;
        double b = pri_seg * (i + 1);
        std::default_random_engine random(time(NULL));
        std::uniform_real_distribution<double> dist(a, b);
        double v = dist(random);
        std::shared_ptr<LeafResult> leafResult = tree->Get_leaf(v);
        result->b_idx[i] = leafResult->leaf_idx;
        double prob = (double)leafResult->p / total_p;
        result->ISWeights[i] = std::pow(prob / min_prob, -beta);
//        sum += result->ISWeights[i];
        result->g_list[i] =leafResult->data->g;
        result->list_st[i] =leafResult->data->s_t;
        result->list_s_primes[i] =leafResult->data->s_prime;
        result->list_at[i] =leafResult->data->a_t;
        result->list_rt[i] =leafResult->data->r_t;
        result->list_term[i] =leafResult->data->term_t;
    }
//    for(int i =0;i<n;++i){
//        result->ISWeights[i] = sum;
//    }
    return result;
}


void Memory::batch_update(std::vector<int> tree_idx, std::vector<double> abs_errors)
{

    for(int i =0;i<(int)tree_idx.size();++i){
        //proportional method, we can also use ranked method here
        abs_errors[i] += epsilon;
        double clipped_error = min(abs_errors[i], abs_err_upper);
        double ps = std::pow(clipped_error, alpha);
        tree->Update(tree_idx[i], ps);
    }
}

