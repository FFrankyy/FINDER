#ifndef DISJOINT_SET_H
#define DISJOINT_SET_H

#include <map>
#include <vector>
#include <memory>
#include <algorithm>
#include <set>
class Disjoint_Set
{
public:
    Disjoint_Set();
    Disjoint_Set(int graphSize);
    ~Disjoint_Set();
    int findRoot(int node);
    void merge(int node1, int node2);
    double getBiggestComponentCurrentRatio() const;
    int getRank(int rootNode) const;
	std::vector<int> unionSet;  //记录每个node所在的连通片的core nodeID
	std::vector<int> rankCount; //记录core nodeID对应的连通片的大小（节点个数）
	int maxRankCount;
	double CCDScore;
};



#endif