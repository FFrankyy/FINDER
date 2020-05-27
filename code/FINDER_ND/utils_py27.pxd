
from libcpp.vector cimport vector
from libcpp.set cimport set
from libcpp.memory cimport shared_ptr
from libcpp cimport bool

cdef extern from "./src/lib/utils_py27.h":
    cdef cppclass Utils:
        Utils()
        double getRobustness(vector[int] edges_from,vector[int] edges_to,int num_nodes,vector[int] solution)except+
        vector[double] MaxWccSzList

