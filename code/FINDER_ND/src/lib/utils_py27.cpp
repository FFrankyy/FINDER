#include "utils_py27.h"
#include <cassert>
#include <random>
#include <algorithm>
#include <set>
#include "stdio.h"
#include <queue>
#include <stack>

Utils::Utils()
{
MaxWccSzList.clear();
}

double Utils::getRobustness(std::vector<int> edges_from, std::vector<int> edges_to,int num_nodes, std::vector<int> solution)
{
    MaxWccSzList.clear();
    std::vector<std::vector<int>> backupCompletedAdjListGraph;
    backupCompletedAdjListGraph.resize(num_nodes);
    for (int i = 0; i < (int)edges_from.size(); ++i)
    {
        int x = edges_from[i];
        int y = edges_to[i];
        backupCompletedAdjListGraph[x].push_back(y);
        backupCompletedAdjListGraph[y].push_back(x);
    }
    std::vector<std::vector<int>> current_adj_list;
    GraphUtil graphutil =GraphUtil();
    Disjoint_Set disjoint_Set =  Disjoint_Set(num_nodes);
    std::vector<bool> backupAllVex(num_nodes, false);
    int totalMaxNum = 0;
    for (std::vector<int>::reverse_iterator it = solution.rbegin(); it != solution.rend(); ++it)
    {
        int Node =(*it);
        graphutil.recoverAddNode(backupCompletedAdjListGraph, backupAllVex, current_adj_list, Node, disjoint_Set);
        totalMaxNum += disjoint_Set.maxRankCount;
        MaxWccSzList.push_back((double)disjoint_Set.maxRankCount / (double)num_nodes);
    }
    totalMaxNum = totalMaxNum - disjoint_Set.maxRankCount;
    std::reverse(MaxWccSzList.begin(), MaxWccSzList.end());
    return (double)totalMaxNum/((double)num_nodes*(double)num_nodes);
}

