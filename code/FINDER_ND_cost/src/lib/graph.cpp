#include "graph.h"
#include <cassert>
#include <iostream>
#include <random>
#include <iterator>
//#include "stdio.h"

Graph::Graph() : num_nodes(0), num_edges(0)
{
    edge_list.clear();
    adj_list.clear();
    nodes_weight.clear();
    total_nodes_weight= 0.0;
}

Graph::Graph(const int _num_nodes, const int _num_edges, const int* edges_from, const int* edges_to,const double* _nodes_weight)
        : num_nodes(_num_nodes), num_edges(_num_edges)
{
    edge_list.resize(num_edges);
    adj_list.resize(num_nodes);
    nodes_weight.clear();
    total_nodes_weight = 0.0;
    for (int i = 0; i < num_nodes; ++i)
    {
        adj_list[i].clear();
        nodes_weight.push_back(_nodes_weight[i]);
        total_nodes_weight+=_nodes_weight[i];
    }

    for (int i = 0; i < num_edges; ++i)
    {
        int x = edges_from[i], y = edges_to[i];
        adj_list[x].push_back(y);
        adj_list[y].push_back(x);
        edge_list[i] = std::make_pair(edges_from[i], edges_to[i]);
    }
//    for (int i = 0; i < num_nodes; ++i)
//        sort(adj_list[i].begin(),adj_list[i].end());
}

double Graph::getTwoRankNeighborsRatio(std::vector<int> covered)
{
    std::set<int> tempSet;
    for(int i =0;i<(int)covered.size();++i){
        tempSet.insert(covered[i]);
    }
    double sum  = 0;
    for(int i =0;i<num_nodes;++i){
        if(tempSet.count(i)==0){
        for(int j=i+1;j<num_nodes;++j){
        if(tempSet.count(j)==0){
            std::vector<int> v3;
            std::set_intersection(adj_list[i].begin(),adj_list[i].end(),adj_list[j].begin(),adj_list[j].end(),std::inserter(v3,v3.begin()));
            if(v3.size()>0){
                sum += 1.0;
            }
        }
        }
        }
    }
    return sum;
}

GSet::GSet()
{
    graph_pool.clear();
}
void GSet::Clear()
{
    graph_pool.clear();
}

void GSet::InsertGraph(int gid, std::shared_ptr<Graph> graph)
{
    assert(graph_pool.count(gid) == 0);

    graph_pool[gid] = graph;
}

std::shared_ptr<Graph> GSet::Get(int gid)
{
    assert(graph_pool.count(gid));
    return graph_pool[gid];
}

std::shared_ptr<Graph> GSet::Sample()
{
//    printf("graph_pool_size:%d",graph_pool.size());
    assert(graph_pool.size());
//    printf("graph_pool_size:%d",graph_pool.size());
    int gid = rand() % graph_pool.size();
    assert(graph_pool[gid]);
    return graph_pool[gid];
}

GSet GSetTrain, GSetTest;