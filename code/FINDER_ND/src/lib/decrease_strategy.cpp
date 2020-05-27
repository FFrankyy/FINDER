#include <vector>
#include <algorithm>
#include <iterator>
#include <unordered_set>
#include <set>
#include "stdio.h"
#include "disjoint_set.h"
#include "math.h"
using namespace std;

class decreaseComponentStrategy
{
public:

	decreaseComponentStrategy()
	{
//		printf("decreaseComponentStrategy:%s\n",description);
	}

	virtual long long decreaseComponentNumIfAddNode(const vector<vector<int> >& backupCompletedAdjListGraph, const vector<bool>& currentAllVex, Disjoint_Set &unionSet, int node) = 0;

};


class decreaseComponentRank :public decreaseComponentStrategy
{

public:
	decreaseComponentRank() :decreaseComponentStrategy(){}

	virtual long long decreaseComponentNumIfAddNode(const vector<vector<int> >& backupCompletedAdjListGraph, const vector<bool>& currentAllVex, Disjoint_Set &unionSet, int node)
	{
		unordered_set<int> componentSet;

		for (int i = 0; i < (int)backupCompletedAdjListGraph[node].size(); i++)
		{
			int neighbourNode = backupCompletedAdjListGraph[node][i];

			if (currentAllVex[neighbourNode])
			{
				componentSet.insert(unionSet.findRoot(neighbourNode));
			}
		}

		long long sum = 1;

		for (int eachNode : componentSet)
		{
			sum += unionSet.getRank(eachNode);
		}
		return sum;
	}
};

class decreaseComponentCount : public decreaseComponentStrategy
{
public:

	decreaseComponentCount() :decreaseComponentStrategy(){}

	virtual long long decreaseComponentNumIfAddNode(const vector<vector<int> >& backupCompletedAdjListGraph, const vector<bool>& currentAllVex, Disjoint_Set &unionSet, int node) override
	{
		unordered_set<int> componentSet;

		for (int i = 0; i < (int)backupCompletedAdjListGraph[node].size(); i++)
		{
			int neighbourNode = backupCompletedAdjListGraph[node][i];

			if (currentAllVex[neighbourNode])
			{
				componentSet.insert(unionSet.findRoot(neighbourNode));
			}
		}

		return (long long)componentSet.size();
	}
};

class decreaseComponentMultiple : public decreaseComponentStrategy
{
public:

	decreaseComponentMultiple() :decreaseComponentStrategy(){}

	virtual long long decreaseComponentNumIfAddNode(const vector<vector<int> >& backupCompletedAdjListGraph, const vector<bool>& currentAllVex, Disjoint_Set &unionSet, int node) override
	{
		unordered_set<int> componentSet;

		for (int i = 0; i < (int)backupCompletedAdjListGraph[node].size(); i++)
		{
			int neighbourNode = backupCompletedAdjListGraph[node][i];

			if (currentAllVex[neighbourNode])
			{
				componentSet.insert(unionSet.findRoot(neighbourNode));
			}
		}

		long long sum = 1;

		for (int eachNode : componentSet)
		{
			sum += unionSet.getRank(eachNode);
		}

		sum *= componentSet.size();

		return sum;
	}
};

