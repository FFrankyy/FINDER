from cython.operator import dereference as deref
from libcpp.memory cimport shared_ptr
import numpy as np
import graph
from graph cimport Graph
import gc
from libc.stdlib cimport free

cdef class py_MvcEnv:
    cdef shared_ptr[MvcEnv] inner_MvcEnv
    cdef shared_ptr[Graph] inner_Graph
    def __cinit__(self,double _norm):
        self.inner_MvcEnv = shared_ptr[MvcEnv](new MvcEnv(_norm))
        self.inner_Graph =shared_ptr[Graph](new Graph())
    # def __dealloc__(self):
    #     if self.inner_MvcEnv != NULL:
    #         self.inner_MvcEnv.reset()
    #         gc.collect()
    #     if self.inner_Graph != NULL:
    #         self.inner_Graph.reset()
    #         gc.collect()
    def s0(self,_g):
        self.inner_Graph =shared_ptr[Graph](new Graph())
        deref(self.inner_Graph).num_nodes = _g.num_nodes
        deref(self.inner_Graph).num_edges = _g.num_edges
        deref(self.inner_Graph).edge_list = _g.edge_list
        deref(self.inner_Graph).adj_list = _g.adj_list
        deref(self.inner_MvcEnv).s0(self.inner_Graph)

    def step(self,int a):
        return deref(self.inner_MvcEnv).step(a)

    def stepWithoutReward(self,int a):
        deref(self.inner_MvcEnv).stepWithoutReward(a)

    def randomAction(self):
        return deref(self.inner_MvcEnv).randomAction()

    def betweenAction(self):
        return deref(self.inner_MvcEnv).betweenAction()

    def isTerminal(self):
        return deref(self.inner_MvcEnv).isTerminal()

    def getReward(self):
        return deref(self.inner_MvcEnv).getReward()

    def getMaxConnectedNodesNum(self):
        return deref(self.inner_MvcEnv).getMaxConnectedNodesNum()

    def getRemainingCNDScore(self):
        return deref(self.inner_MvcEnv).getRemainingCNDScore()

    @property
    def norm(self):
        return deref(self.inner_MvcEnv).norm

    @property
    def graph(self):
        # temp_innerGraph=deref(self.inner_Graph)   #得到了Graph 对象
        return self.G2P(deref(self.inner_Graph))

    @property
    def state_seq(self):
        return deref(self.inner_MvcEnv).state_seq

    @property
    def act_seq(self):
        return deref(self.inner_MvcEnv).act_seq

    @property
    def action_list(self):
        return deref(self.inner_MvcEnv).action_list

    @property
    def reward_seq(self):
        return deref(self.inner_MvcEnv).reward_seq

    @property
    def sum_rewards(self):
        return deref(self.inner_MvcEnv).sum_rewards

    @property
    def numCoveredEdges(self):
        return deref(self.inner_MvcEnv).numCoveredEdges

    @property
    def covered_set(self):
        return deref(self.inner_MvcEnv).covered_set

    @property
    def avail_list(self):
        return deref(self.inner_MvcEnv).avail_list


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


    # cdef reshape_Graph(self, int _num_nodes, int _num_edges, int[:] edges_from, int[:] edges_to):
    #     cdef int *cint_edges_from = <int*>malloc(_num_edges*sizeof(int))
    #     cdef int *cint_edges_to = <int*>malloc(_num_edges*sizeof(int))
    #     cdef int i
    #     for i in range(_num_edges):
    #         cint_edges_from[i] = edges_from[i]
    #     for i in range(_num_edges):
    #         cint_edges_to[i] = edges_to[i]
    #     free(cint_edges_from)
    #     free(cint_edges_to)
    #     return  new Graph(_num_nodes,_num_edges,&cint_edges_from[0],&cint_edges_to[0])