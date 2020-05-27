#include "graph_utils.h"
#include <cassert>
#include <iostream>
#include <random>
#include <iterator>
//#include "stdio.h"

GraphUtil::GraphUtil()
{

}

GraphUtil::~GraphUtil()
{

}


 void GraphUtil::deleteNode(std::vector<std::vector<int> > &adjListGraph, int node)
{
//    for (auto neighbour : adjListGraph[node])
//    {
//        adjListGraph[neighbour].erase(remove(adjListGraph[neighbour].begin(), adjListGraph[neighbour].end(), node), adjListGraph[neighbour].end());
//    }
    for (int i=0; i<(int)adjListGraph[node].size(); ++i)
    {
        int neighbour = adjListGraph[node][i];
        adjListGraph[neighbour].erase(remove(adjListGraph[neighbour].begin(), adjListGraph[neighbour].end(), node), adjListGraph[neighbour].end());
    }
    adjListGraph[node].clear();
}

void GraphUtil::recoverAddNode(const std::vector<std::vector<int> > &backupCompletedAdjListGraph, std::vector<bool>& backupAllVex, std::vector<std::vector<int> > &adjListGraph, int node, Disjoint_Set &unionSet)
{

    for (int i = 0; i < (int)backupCompletedAdjListGraph[node].size(); i++)
    {
        int neighbourNode = backupCompletedAdjListGraph[node][i];

        if (backupAllVex[neighbourNode])
        {
            addEdge(adjListGraph, node, neighbourNode);
            unionSet.merge(node, neighbourNode);
        }
    }

    backupAllVex[node] = true;
}

void GraphUtil::addEdge(std::vector<std::vector<int> > &adjListGraph, int node0, int node1)
{
    if (((int)adjListGraph.size() - 1) < std::max(node0, node1))
    {
        adjListGraph.resize(std::max(node0, node1) + 1);
    }

    adjListGraph[node0].push_back(node1);
    adjListGraph[node1].push_back(node0);
}

