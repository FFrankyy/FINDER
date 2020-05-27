from cython.operator import dereference as deref
from libcpp.memory cimport shared_ptr
import numpy as np
import graph
from libc.stdlib cimport malloc
from libc.stdlib cimport free
from graph cimport Graph
import tensorflow as tf
from scipy.sparse import coo_matrix
# import gc


cdef class py_sparseMatrix:
    cdef shared_ptr[sparseMatrix] inner_sparseMatrix
    def __cinit__(self):
        self.inner_sparseMatrix =shared_ptr[sparseMatrix](new sparseMatrix())
    # def __dealloc__(self):
    #     if self.inner_sparseMatrix != NULL:
    #         self.inner_sparseMatrix.reset()
    #         gc.collect()
    @property
    def rowIndex(self):
        return deref(self.inner_sparseMatrix).rowIndex
    @property
    def colIndex(self):
        return deref(self.inner_sparseMatrix).colIndex
    @property
    def value(self):
        return deref(self.inner_sparseMatrix).value
    @property
    def rowNum(self):
        return deref(self.inner_sparseMatrix).rowNum
    @property
    def colNum(self):
        return deref(self.inner_sparseMatrix).colNum


cdef class py_PrepareBatchGraph:
    cdef shared_ptr[PrepareBatchGraph] inner_PrepareBatchGraph
    cdef sparseMatrix matrix
    def __cinit__(self,aggregatorID):
        self.inner_PrepareBatchGraph =shared_ptr[PrepareBatchGraph](new PrepareBatchGraph(aggregatorID))
    # def __dealloc__(self):
    #     if self.inner_PrepareBatchGraph != NULL:
    #         self.inner_PrepareBatchGraph.reset()
    #         gc.collect()
    def SetupTrain(self,idxes,g_list,covered,list actions):
        cdef shared_ptr[Graph] inner_Graph
        cdef vector[shared_ptr[Graph]] inner_glist
        for _g in g_list:
            # inner_glist.push_back(_g.inner_Graph)
            inner_Graph = shared_ptr[Graph](new Graph())
            deref(inner_Graph).num_nodes = _g.num_nodes
            deref(inner_Graph).num_edges = _g.num_edges
            deref(inner_Graph).edge_list = _g.edge_list
            deref(inner_Graph).adj_list = _g.adj_list
            inner_glist.push_back(inner_Graph)

        cdef int *refint = <int*>malloc(len(actions)*sizeof(int))
        cdef int i
        for i in range(len(actions)):
            refint[i] = actions[i]
        deref(self.inner_PrepareBatchGraph).SetupTrain(idxes,inner_glist,covered,refint)
        free(refint)

    def SetupPredAll(self,idxes,g_list,covered):
        cdef shared_ptr[Graph] inner_Graph
        cdef vector[shared_ptr[Graph]] inner_glist
        for _g in g_list:
            # inner_glist.push_back(_g.inner_Graph)
            inner_Graph = shared_ptr[Graph](new Graph())
            deref(inner_Graph).num_nodes = _g.num_nodes
            deref(inner_Graph).num_edges = _g.num_edges
            deref(inner_Graph).edge_list = _g.edge_list
            deref(inner_Graph).adj_list = _g.adj_list
            inner_glist.push_back(inner_Graph)
        deref(self.inner_PrepareBatchGraph).SetupPredAll(idxes,inner_glist,covered)

    @property
    def act_select(self):
        self.matrix = deref(deref(self.inner_PrepareBatchGraph).act_select)
        return self.ConvertSparseToTensor(self.matrix)
    @property
    def rep_global(self):
        matrix = deref(deref(self.inner_PrepareBatchGraph).rep_global)
        return self.ConvertSparseToTensor(matrix)
        #return coo_matrix((data, (rowIndex,colIndex)), shape=(rowNum,colNum))
    @property
    def n2nsum_param(self):
        matrix = deref(deref(self.inner_PrepareBatchGraph).n2nsum_param)
        return self.ConvertSparseToTensor(matrix)
    @property
    def laplacian_param(self):
        matrix = deref(deref(self.inner_PrepareBatchGraph).laplacian_param)
        return self.ConvertSparseToTensor(matrix)
    @property
    def subgsum_param(self):
        matrix = deref(deref(self.inner_PrepareBatchGraph).subgsum_param)
        return self.ConvertSparseToTensor(matrix)
    @property
    def idx_map_list(self):
        return deref(self.inner_PrepareBatchGraph).idx_map_list
    @property
    def subgraph_id_span(self):
        return deref(self.inner_PrepareBatchGraph).subgraph_id_span
    @property
    def aux_feat(self):
        return deref(self.inner_PrepareBatchGraph).aux_feat
    @property
    def aggregatorID(self):
        return deref(self.inner_PrepareBatchGraph).aggregatorID
    @property
    def avail_act_cnt(self):
        return deref(self.inner_PrepareBatchGraph).avail_act_cnt

    cdef ConvertSparseToTensor(self,sparseMatrix matrix):

        rowIndex= matrix.rowIndex
        colIndex= matrix.colIndex
        data= matrix.value
        rowNum= matrix.rowNum
        colNum= matrix.colNum
        indices = np.mat([rowIndex, colIndex]).transpose()
        return tf.SparseTensorValue(indices, data, (rowNum,colNum))




