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
	std::vector<int> unionSet;
	std::vector<int> rankCount;
	int maxRankCount;
    double CCDScore;
};



#endif
