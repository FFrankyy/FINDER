from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr
from libcpp cimport bool
from graph cimport Graph
from mvc_env cimport MvcEnv


cdef extern from "./src/lib/nstep_replay_mem_prioritized.h":
    cdef cppclass Data:
        Data()except+
        shared_ptr[Graph] g
        vector[int] s_t
        vector[int] s_prime
        int a_t
        double r_t
        bool term_t

cdef extern from "./src/lib/nstep_replay_mem_prioritized.h":
    cdef cppclass LeafResult:
        LeafResult()except+
        int leaf_idx
        double p
        shared_ptr[Data] data

cdef extern from "./src/lib/nstep_replay_mem_prioritized.h":
    cdef cppclass SumTree:
        SumTree(int capacity)except+
        int data_pointer
        int capacity
        vector[double] tree
        vector[shared_ptr[Data]] data
        void Add(double p, shared_ptr[Data])
        void Update(int tree_idx,double p)
        shared_ptr[LeafResult] Get_leaf(double v)


cdef extern from "./src/lib/nstep_replay_mem_prioritized.h":
    cdef cppclass ReplaySample:
        ReplaySample(int batch_size)except+
        vector[int] b_idx
        vector[double] ISWeights
        vector[shared_ptr[Graph]] g_list
        vector[vector[int]] list_st
        vector[vector[int]] list_s_primes
        vector[int] list_at
        vector[double] list_rt
        vector[bool] list_term


cdef extern from "./src/lib/nstep_replay_mem_prioritized.h":
    cdef cppclass Memory:
        Memory(double epsilon,double alpha,double beta,double beta_increment_per_sampling,double abs_err_upper,int capacity)except+
        shared_ptr[SumTree] tree
        double epsilon
        double alpha
        double beta
        double beta_increment_per_sampling
        double abs_err_upper
        void Store(shared_ptr[Data] transition)except+
        void Add(shared_ptr[MvcEnv] env,int n_step)except+
        shared_ptr[ReplaySample] Sampling(int n)except+
        void batch_update(vector[int] tree_idx, vector[double] abs_errors)except+
