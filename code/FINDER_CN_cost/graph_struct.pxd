from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.pair cimport pair
cdef extern from "./src/lib/graph_struct.h":
    cdef cppclass LinkedTable[T]:
        LinkedTable()except+
        void AddEntry(int head_id, T content)except+
        void Resize(int new_n)except+
        int n
        vector[vector[T]] head
    cdef cppclass GraphStruct:
        GraphStruct()except+
        void AddEdge(int idx, int x, int y)except+
        void AddNode(int subg_id, int n_idx)except+
        void Resize(unsigned _num_subgraph, unsigned _num_nodes)except+
        LinkedTable[pair[int, int]] *out_edges
        LinkedTable[pair[int, int]] *in_edges
        LinkedTable[int]* subgraph
        vector[pair[int, int]] edge_list
        unsigned num_nodes
        unsigned num_edges
        unsigned num_subgraph
