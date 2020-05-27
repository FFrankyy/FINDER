from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr
from libcpp.pair cimport pair
from graph cimport Graph
cdef extern from "./src/lib/PrepareBatchGraph.h":
    cdef cppclass sparseMatrix:
        sparseMatrix()except+
        vector[int] rowIndex
        vector[int] colIndex
        vector[double] value
        int rowNum
        int colNum

    cdef cppclass PrepareBatchGraph:
        PrepareBatchGraph(int aggregatorID)except+
        void SetupTrain(vector[int] idxes,vector[shared_ptr[Graph] ] g_list,vector[vector[int]] covered,const int* actions)except+
        void SetupPredAll(vector[int] idxes,vector[shared_ptr[Graph] ] g_list,vector[vector[int]] covered)except+
        shared_ptr[sparseMatrix] act_select
        shared_ptr[sparseMatrix] rep_global
        shared_ptr[sparseMatrix] n2nsum_param
        shared_ptr[sparseMatrix] laplacian_param
        shared_ptr[sparseMatrix] subgsum_param
        vector[vector[int]]  idx_map_list
        vector[pair[int,int]] subgraph_id_span
        vector[vector[double]]  aux_feat
        vector[int] avail_act_cnt
        int aggregatorID