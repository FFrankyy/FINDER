from cython.operator import dereference as deref
from libcpp.memory cimport shared_ptr
# import gc
from libc.stdlib cimport free

cdef class py_GraphStruct:
    cdef shared_ptr[GraphStruct] inner_GraphStruct
    def __cinit__(self):
        self.inner_GraphStruct = shared_ptr[GraphStruct](new GraphStruct())
    # def __dealloc__(self):
    #     if self.inner_GraphStruct != NULL:
    #         self.inner_GraphStruct.reset()
    #         gc.collect()
    def AddEdge(self,int idx, int x, int y):
        deref(self.inner_GraphStruct).AddEdge(idx,x,y)

    def AddNode(self,int subg_id, int n_idx):
        deref(self.inner_GraphStruct).AddNode(subg_id,n_idx)

    def Resize(self,unsigned _num_subgraph, unsigned _num_nodes):
        deref(self.inner_GraphStruct).Resize(_num_subgraph,_num_nodes)

    @property
    def num_nodes(self):
        return deref(self.inner_GraphStruct).num_nodes

    @property
    def num_edges(self):
        return deref(self.inner_GraphStruct).num_edges

    @property
    def num_subgraph(self):
        return deref(self.inner_GraphStruct).num_subgraph

    @property
    def edge_list(self):
        return deref(self.inner_GraphStruct).edge_list

    @property
    def out_edges(self):
        return deref(deref(self.inner_GraphStruct).out_edges).head
    @property
    def in_edges(self):
        return deref(deref(self.inner_GraphStruct).in_edges).head
        #     cint_edges_from = np.zeros([num_edges],dtype=np.int)
        # cint_edges_to = np.zeros([num_edges],dtype=np.int)
        # for i in range(num_edges):
        #     cint_edges_from[i]=edge_list[i].first
        #     cint_edges_to[i] =edge_list[i].second