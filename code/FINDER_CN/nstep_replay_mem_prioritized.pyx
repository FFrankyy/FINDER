from cython.operator import dereference as deref
from libcpp.memory cimport shared_ptr
from libc.stdlib cimport malloc
from libcpp cimport bool
import  graph
import  numpy as np
# import gc
from libc.stdlib cimport free


cdef class py_Data:
    cdef shared_ptr[Data] inner_Data#使用unique_ptr优于shared_ptr
    def __cinit__(self):
        self.inner_Data = shared_ptr[Data](new Data())
    # def __dealloc__(self):
    #     if self.inner_Data != NULL:
    #         self.inner_Data.reset()
    #         gc.collect()
    @property
    def graph(self):
        return self.G2P(deref(deref(self.inner_Data).g))
    @property
    def s_t(self):
        return deref(self.inner_Data).s_t
    @property
    def a_t(self):
        return deref(self.inner_Data).a_t
    @property
    def r_t(self):
        return deref(self.inner_Data).r_t
    @property
    def s_prime(self):
        return deref(self.inner_Data).s_prime
    @property
    def term_t(self):
        return deref(self.inner_Data).term_t

    cdef G2P(self,Graph graph1):
        num_nodes = graph1.num_nodes     #得到Graph对象的节点个数
        num_edges = graph1.num_edges    #得到Graph对象的连边个数
        edge_list = graph1.edge_list
        cint_edges_from = np.zeros([num_edges],dtype=np.int)
        cint_edges_to = np.zeros([num_edges],dtype=np.int)
        for i in range(num_edges):
            cint_edges_from[i]=edge_list[i].first
            cint_edges_to[i] =edge_list[i].second
        return graph.py_Graph(num_nodes,num_edges,cint_edges_from,cint_edges_to)

cdef class py_LeafResult:
    cdef shared_ptr[LeafResult] inner_LeafResult
    cdef shared_ptr[Data] inner_Data
    def __cinit__(self):
        self.inner_LeafResult = shared_ptr[LeafResult](new LeafResult())
    # def __dealloc__(self):
    #     if self.inner_LeafResult != NULL:
    #         self.inner_LeafResult.reset()
    #         gc.collect()
    @property
    def leaf_idx(self):
         return deref(self.inner_LeafResult).leaf_idx
    @property
    def p(self):
        return deref(self.inner_LeafResult).p
    @property
    def Data(self):
        self.inner_Data =  deref(self.inner_LeafResult).data
        result = py_Data()
        result.inner_Data = self.inner_Data
        return result

cdef class py_SumTree:
    cdef shared_ptr[SumTree] inner_SumTree
    cdef shared_ptr[Data] inner_Data
    cdef shared_ptr[Graph] inner_Graph
    cdef shared_ptr[LeafResult] inner_LeafResult
    def __cinit__(self,capacity):
        self.inner_SumTree = shared_ptr[SumTree](new SumTree(capacity))
    # def __dealloc__(self):
    #     if self.inner_SumTree != NULL:
    #         self.inner_SumTree.reset()
    #         gc.collect()

    @property
    def capacity(self):
        return deref(self.inner_SumTree).capacity
    @property
    def data_pointer(self):
         return deref(self.inner_SumTree).data_pointer
    @property
    def tree(self):
        return deref(self.inner_SumTree).tree
    @property
    def data(self):
        result = []
        for dataPtr in deref(self.inner_SumTree).data:
            Data = py_Data()
            Data.inner_Data = dataPtr
            result.append(Data)
        return  result
    def Add(self,p,pyData):
        g = pyData.graph
        self.inner_Data =shared_ptr[Data](new Data())
        self.inner_Graph =shared_ptr[Graph](new Graph())
        deref(self.inner_Graph).num_nodes= g.num_nodes
        deref(self.inner_Graph).num_edges=g.num_edges
        deref(self.inner_Graph).edge_list=g.edge_list
        deref(self.inner_Graph).adj_list=g.adj_list
        deref(self.inner_Data).g=self.inner_Graph
        deref(self.inner_Data).s_t= pyData.s_t
        deref(self.inner_Data).a_t= pyData.a_t
        deref(self.inner_Data).r_t= pyData.r_t
        deref(self.inner_Data).s_prime= pyData.s_prime
        deref(self.inner_Data).term_t= pyData.term_t
        deref(self.inner_SumTree).Add(p,self.inner_Data)

    def Update(self,int tree_idx,double p):
        deref(self.inner_SumTree).Update(tree_idx,p)

    def Get_leaf(self,double v):
        self.inner_LeafResult = deref(self.inner_SumTree).Get_leaf(v)
        result = py_LeafResult()
        result.inner_LeafResult = self.inner_LeafResult
        return result

cdef class py_ReplaySample:
    cdef shared_ptr[ReplaySample] inner_ReplaySample
    def __cinit__(self,int n):
        self.inner_ReplaySample = shared_ptr[ReplaySample](new ReplaySample(n))
    # def __dealloc__(self):
    #     if self.inner_ReplaySample != NULL:
    #         self.inner_ReplaySample.reset()
    #         gc.collect()
    @property
    def b_idx(self):
        return deref(self.inner_ReplaySample).b_idx
    @property
    def ISWeights(self):
        return deref(self.inner_ReplaySample).ISWeights
    @property
    def g_list(self):
        result = []
        for graphPtr in deref(self.inner_ReplaySample).g_list:
            result.append(self.G2P(deref(graphPtr)))
        return  result
    @property
    def list_st(self):
        return deref(self.inner_ReplaySample).list_st
    @property
    def list_s_primes(self):
        return deref(self.inner_ReplaySample).list_s_primes
    @property
    def list_at(self):
        return deref(self.inner_ReplaySample).list_at
    @property
    def list_rt(self):
        return deref(self.inner_ReplaySample).list_rt
    @property
    def list_term(self):
        return deref(self.inner_ReplaySample).list_term

    cdef G2P(self,Graph graph1):
        num_nodes = graph1.num_nodes     #得到Graph对象的节点个数
        num_edges = graph1.num_edges    #得到Graph对象的连边个数
        edge_list = graph1.edge_list
        cint_edges_from = np.zeros([num_edges],dtype=np.int)
        cint_edges_to = np.zeros([num_edges],dtype=np.int)
        for i in range(num_edges):
            cint_edges_from[i]=edge_list[i].first
            cint_edges_to[i] =edge_list[i].second
        return graph.py_Graph(num_nodes,num_edges,cint_edges_from,cint_edges_to)



cdef class py_Memory:
    cdef shared_ptr[Memory] inner_Memory
    cdef shared_ptr[SumTree] inner_SumTree
    cdef shared_ptr[Data] inner_Data
    cdef shared_ptr[ReplaySample] inner_ReplaySample
    cdef shared_ptr[Graph] inner_Graph
    cdef shared_ptr[MvcEnv] inner_MvcEnv
    def __cinit__(self,double epsilon,double alpha,double beta,double beta_increment_per_sampling,double abs_err_upper,int capacity):
        self.inner_Memory = shared_ptr[Memory](new Memory(epsilon,alpha,beta,beta_increment_per_sampling,abs_err_upper,capacity))
    # def __dealloc__(self):
    #     if self.inner_Memory != NULL:
    #         self.inner_Memory.reset()
    #         gc.collect()
    #     if self.inner_SumTree != NULL:
    #         self.inner_SumTree.reset()
    #         gc.collect()
    #     if self.inner_Data != NULL:
    #         self.inner_Data.reset()
    #         gc.collect()
    #     if self.inner_ReplaySample != NULL:
    #         self.inner_ReplaySample.reset()
    #         gc.collect()
    #     if self.inner_Graph != NULL:
    #         self.inner_Graph.reset()
    #         gc.collect()
    #     if self.inner_MvcEnv != NULL:
    #         self.inner_MvcEnv.reset()
    #         gc.collect()
    @property
    def tree(self):
        self.inner_SumTree =  deref(self.inner_Memory).tree
        result = py_SumTree()
        result.inner_SumTree = self.inner_SumTree
        return result
    @property
    def epsilon(self):
        return deref(self.inner_Memory).epsilon
    @property
    def alpha(self):
        return deref(self.inner_Memory).alpha
    @property
    def beta(self):
        return deref(self.inner_Memory).beta
    @property
    def beta_increment_per_sampling(self):
        return deref(self.inner_Memory).beta_increment_per_sampling
    @property
    def abs_err_upper(self):
        return deref(self.inner_Memory).abs_err_upper

    def Store(self,transition):
        g = transition.graph
        self.inner_Data =shared_ptr[Data](new Data())
        self.inner_Graph =shared_ptr[Graph](new Graph())
        deref(self.inner_Graph).num_nodes= g.num_nodes
        deref(self.inner_Graph).num_edges=g.num_edges
        deref(self.inner_Graph).edge_list=g.edge_list
        deref(self.inner_Graph).adj_list=g.adj_list
        deref(self.inner_Data).g=self.inner_Graph
        deref(self.inner_Data).s_t= transition.s_t
        deref(self.inner_Data).a_t=transition.a_t
        deref(self.inner_Data).r_t=transition.r_t
        deref(self.inner_Data).s_prime=transition.s_prime
        deref(self.inner_Data).term_t=transition.term_t
        deref(self.inner_Memory).Store(self.inner_Data)

    def Add(self,mvcenv,int nstep):
        self.inner_Graph =shared_ptr[Graph](new Graph())
        # g = self.GenNetwork(mvcenv.graph)
        g = mvcenv.graph
        deref(self.inner_Graph).num_nodes= g.num_nodes
        deref(self.inner_Graph).num_edges=g.num_edges
        deref(self.inner_Graph).edge_list=g.edge_list
        deref(self.inner_Graph).adj_list=g.adj_list
        self.inner_MvcEnv = shared_ptr[MvcEnv](new MvcEnv(mvcenv.norm))
        deref(self.inner_MvcEnv).norm = mvcenv.norm
        deref(self.inner_MvcEnv).graph = self.inner_Graph
        deref(self.inner_MvcEnv).state_seq = mvcenv.state_seq
        deref(self.inner_MvcEnv).act_seq = mvcenv.act_seq
        deref(self.inner_MvcEnv).action_list = mvcenv.action_list
        deref(self.inner_MvcEnv).reward_seq = mvcenv.reward_seq
        deref(self.inner_MvcEnv).sum_rewards = mvcenv.sum_rewards
        deref(self.inner_MvcEnv).numCoveredEdges = mvcenv.numCoveredEdges
        deref(self.inner_MvcEnv).covered_set = mvcenv.covered_set
        deref(self.inner_MvcEnv).avail_list = mvcenv.avail_list
        deref(self.inner_Memory).Add(self.inner_MvcEnv,nstep)

    def Sampling(self,int n):
        self.inner_ReplaySample=deref(self.inner_Memory).Sampling(n)
        result = py_ReplaySample(n)
        result.inner_ReplaySample = self.inner_ReplaySample
        return  result

    def batch_update(self,tree_idx,abs_errors):
        deref(self.inner_Memory).batch_update(tree_idx,abs_errors)

