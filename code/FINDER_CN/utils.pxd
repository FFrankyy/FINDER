
from libcpp.vector cimport vector
from libcpp.set cimport set
from libcpp.memory cimport shared_ptr
from libcpp cimport bool
from graph cimport Graph

cdef extern from "./src/lib/utils.h":
    cdef cppclass Utils:
        Utils()
        double getRobustness(shared_ptr[Graph] graph,vector[int] solution)except+
        vector[int] reInsert(shared_ptr[Graph] graph,vector[int] solution,vector[int] allVex,int decreaseStrategyID,int reinsertEachStep)except+
        int getMxWccSz(shared_ptr[Graph] graph)
        vector[double] Betweenness(shared_ptr[Graph] graph)
        vector[double] MaxWccSzList

