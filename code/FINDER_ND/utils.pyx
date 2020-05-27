from cython.operator import dereference as deref
from libcpp.memory cimport shared_ptr
import numpy as np
import graph
from graph cimport Graph
import gc
from libc.stdlib cimport free

cdef class py_Utils:
    cdef shared_ptr[Utils] inner_Utils
    cdef shared_ptr[Graph] inner_Graph
    def __cinit__(self):
        self.inner_Utils = shared_ptr[Utils](new Utils())
    # def __dealloc__(self):
    #     if self.inner_Utils != NULL:
    #         self.inner_Utils.reset()
    #         gc.collect()

    def getRobustness(self,_g,solution):
        self.inner_Graph =shared_ptr[Graph](new Graph())
        deref(self.inner_Graph).num_nodes = _g.num_nodes
        deref(self.inner_Graph).num_edges = _g.num_edges
        deref(self.inner_Graph).edge_list = _g.edge_list
        deref(self.inner_Graph).adj_list = _g.adj_list
        return deref(self.inner_Utils).getRobustness(self.inner_Graph,solution)


    def reInsert(self,_g,solution,allVex,int decreaseStrategyID,int reinsertEachStep):
        self.inner_Graph =shared_ptr[Graph](new Graph())
        deref(self.inner_Graph).num_nodes = _g.num_nodes
        deref(self.inner_Graph).num_edges = _g.num_edges
        deref(self.inner_Graph).edge_list = _g.edge_list
        deref(self.inner_Graph).adj_list = _g.adj_list
        return deref(self.inner_Utils).reInsert(self.inner_Graph,solution,allVex,decreaseStrategyID,reinsertEachStep)

    def getMxWccSz(self, _g):
        self.inner_Graph =shared_ptr[Graph](new Graph())
        deref(self.inner_Graph).num_nodes = _g.num_nodes
        deref(self.inner_Graph).num_edges = _g.num_edges
        deref(self.inner_Graph).edge_list = _g.edge_list
        deref(self.inner_Graph).adj_list = _g.adj_list
        return deref(self.inner_Utils).getMxWccSz(self.inner_Graph)

    def Betweenness(self,_g):
        self.inner_Graph =shared_ptr[Graph](new Graph())
        deref(self.inner_Graph).num_nodes = _g.num_nodes
        deref(self.inner_Graph).num_edges = _g.num_edges
        deref(self.inner_Graph).edge_list = _g.edge_list
        deref(self.inner_Graph).adj_list = _g.adj_list
        return deref(self.inner_Utils).Betweenness(self.inner_Graph)

    @property
    def MaxWccSzList(self):
        return deref(self.inner_Utils).MaxWccSzList
