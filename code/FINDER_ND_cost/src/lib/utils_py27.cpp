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

double Utils::getRobustness(std::vector<int> edges_from, std::vector<int> edges_to,int num_nodes,std::vector<double> nodes_weight, std::vector<int> solution)
{
    MaxWccSzList.clear();
    std::vector<std::vector<int>> backupCompletedAdjListGraph;
    backupCompletedAdjListGraph.resize(num_nodes);
    for (int i = 0; i < (int)edges_from.size(); ++i)
    {
        int x = edges_from[i], y = edges_to[i];
        backupCompletedAdjListGraph[x].push_back(y);
        backupCompletedAdjListGraph[y].push_back(x);
    }
    std::vector<std::vector<int>> current_adj_list;
    GraphUtil graphutil =GraphUtil();
    Disjoint_Set disjoint_Set =  Disjoint_Set(num_nodes);
    std::vector<bool> backupAllVex(num_nodes, false);
    int totalMaxNum = 0;
    double solution_weights = 0.0;
    double total_nodes_weight =0.0;
    for(int i =0;i<(int)nodes_weight.size();++i){
        total_nodes_weight+=nodes_weight[i];
    }


    for (int i = (int)solution.size()-1;i>=0;i=i-1)
    {
        int Node = solution[i];
        if (i == 0)
         {
            solution_weights = 0;
         }
         else
         {
            solution_weights = nodes_weight[solution[i-1]];
         }
        graphutil.recoverAddNode(backupCompletedAdjListGraph, backupAllVex, current_adj_list, Node, disjoint_Set);
        totalMaxNum += (double)disjoint_Set.maxRankCount * solution_weights;
        MaxWccSzList.push_back((double)disjoint_Set.maxRankCount /num_nodes);
    }

    std::reverse(MaxWccSzList.begin(), MaxWccSzList.end());

    return (double)totalMaxNum/(num_nodes*total_nodes_weight);
}

