#include "graph.h"
#include <cassert>
#include <iostream>
#include <random>
#include <iterator>
#include "disjoint_set.h"
//#include "stdio.h"

Disjoint_Set::Disjoint_Set(int graphSize)
{
    unionSet.resize(graphSize);
    rankCount.resize(graphSize);
    for (int i = 0; i < (int)unionSet.size(); i++)
    {
        unionSet[i] = i;
        rankCount[i] = 1;
    }
    maxRankCount = 1;
    CCDScore = 0.0;
}

Disjoint_Set::~Disjoint_Set()
{
    unionSet.clear();
    rankCount.clear();
    maxRankCount = 1;
    CCDScore = 0.0;
}

int Disjoint_Set::findRoot(int node)
{
    if (node != unionSet[node])
    {
        int rootNode = findRoot(unionSet[node]);
        unionSet[node] = rootNode;
        return rootNode;
    }
    else
    {
        return node;
    }
}

void Disjoint_Set::merge(int node1, int node2)
{
    int node1Root = findRoot(node1);
    int node2Root = findRoot(node2);
    if (node1Root != node2Root)
    {
        double node1Rank = (double)rankCount[node1Root];
        double node2Rank = (double)rankCount[node2Root];
        CCDScore = CCDScore - node1Rank*(node1Rank-1)/2.0 - node2Rank*(node2Rank-1)/2.0;
        CCDScore = CCDScore + (node1Rank+node2Rank)*(node1Rank+node2Rank-1)/2.0;

        if (rankCount[node2Root] > rankCount[node1Root])
        {
            unionSet[node1Root] = node2Root;
            rankCount[node2Root] += rankCount[node1Root];

            if (rankCount[node2Root] > maxRankCount)
            {
                maxRankCount = rankCount[node2Root];
            }
        }
        else
        {
            unionSet[node2Root] = node1Root;
            rankCount[node1Root] += rankCount[node2Root];

            if (rankCount[node1Root] > maxRankCount)
            {
                maxRankCount = rankCount[node1Root];
            }

        }
    }
}


double Disjoint_Set::getBiggestComponentCurrentRatio() const
{
    return double(maxRankCount) / double(rankCount.size());
}


int Disjoint_Set::getRank(int rootNode) const
{
    return rankCount[rootNode];
}

