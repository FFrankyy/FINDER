#ifndef GRAPH_UTILS_H
#define GRAPH_UTILS_H

#include <map>
#include <vector>
#include <memory>
#include <algorithm>
#include <set>
#include "disjoint_set.h"


class GraphUtil
{
public:
    GraphUtil();

    ~GraphUtil();

    void deleteNode(std::vector<std::vector<int> > &adjListGraph, int node);

     void recoverAddNode(const std::vector<std::vector<int> > &backupCompletedAdjListGraph, std::vector<bool>& backupAllVex, std::vector<std::vector<int> > &adjListGraph, int node, Disjoint_Set &unionSet);

     void addEdge(std::vector<std::vector<int> > &adjListGraph, int node0, int node1);


};



#endif