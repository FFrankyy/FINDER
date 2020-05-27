#ifndef GRAPH_H
#define GRAPH_H

#include <map>
#include <vector>
#include <memory>
#include <algorithm>
#include <set>
class Graph
{
public:
    Graph();
    Graph(const int _num_nodes, const int _num_edges, const int* edges_from, const int* edges_to,const double* _nodes_weight);
    int num_nodes;
    int num_edges;
    std::vector< std::vector< int > > adj_list;
    std::vector< std::pair<int, int> > edge_list;
    std::vector<double> nodes_weight;
    double total_nodes_weight;
    double getTwoRankNeighborsRatio(std::vector<int> covered);

};

class GSet
{
public:
    GSet();
    void InsertGraph(int gid, std::shared_ptr<Graph> graph);
    std::shared_ptr<Graph> Sample();
    std::shared_ptr<Graph> Get(int gid);
    void Clear();
    std::map<int, std::shared_ptr<Graph> > graph_pool;
};

extern GSet GSetTrain;
extern GSet GSetTest;

#endif