import numpy as np
from libc.stdlib cimport free

cdef class py_Utils:
    cdef Utils inner_Utils
    def __cinit__(self):
        self.inner_Utils = Utils()


    def getRobustness(self,edges_from,edges_to,num_nodes,nodes_weight,solution):
        return self.inner_Utils.getRobustness(edges_from,edges_to,num_nodes,nodes_weight,solution)


    @property
    def MaxWccSzList(self):
        return self.inner_Utils.MaxWccSzList
