#include "utils.h"
#include "graph.h"
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


std::vector<int> Utils::reInsert(std::shared_ptr<Graph> graph,std::vector<int> solution,const std::vector<int> allVex,int decreaseStrategyID,int reinsertEachStep){
    std::shared_ptr<decreaseComponentStrategy> decreaseStrategy;

    switch(decreaseStrategyID){
        case 1:{
            decreaseStrategy = std::shared_ptr<decreaseComponentStrategy>(new decreaseComponentCount());
            break;
        }
        case 2:{
            decreaseStrategy = std::shared_ptr<decreaseComponentStrategy>(new decreaseComponentRank());
            break;
        }
        case 3:{
            decreaseStrategy = std::shared_ptr<decreaseComponentStrategy>(new decreaseComponentMultiple());
            break;
        }
        default:
        {
            decreaseStrategy = std::shared_ptr<decreaseComponentStrategy>(new decreaseComponentRank());
            break;
        }
    }

        return reInsert_inner(solution,graph,allVex,decreaseStrategy,reinsertEachStep);

}


std::vector<int> Utils::reInsert_inner(const std::vector<int> &beforeOutput, std::shared_ptr<Graph> &graph, const std::vector<int> &allVex, std::shared_ptr<decreaseComponentStrategy> &decreaseStrategy,int reinsertEachStep)
{
    std::shared_ptr<GraphUtil> graphutil =std::shared_ptr<GraphUtil>(new GraphUtil());

    std::vector<std::vector<int> > currentAdjListGraph;

    std::vector<std::vector<int>> backupCompletedAdjListGraph = graph->adj_list;

    std::vector<bool> currentAllVex(graph->num_nodes, false);

    for (int eachV : allVex)
    {
        currentAllVex[eachV] = true;
    }

    std::unordered_set<int> leftOutput(beforeOutput.begin(), beforeOutput.end());

    std::vector<int> finalOutput;

    Disjoint_Set disjoint_Set =  Disjoint_Set(graph->num_nodes);

    while (leftOutput.size() != 0)
    {
//        printf (" reInsertCount:%d\n", leftOutput.size());

        std::vector<std::pair<long long, int> >  batchList;

        for (int eachNode : leftOutput)
        {
            //min is better
            long long decreaseValue = decreaseStrategy->decreaseComponentNumIfAddNode(backupCompletedAdjListGraph, currentAllVex, disjoint_Set, eachNode);
            batchList.push_back(make_pair(decreaseValue, eachNode));
        }


        if (reinsertEachStep >= (int)batchList.size())
        {
            reinsertEachStep = (int)batchList.size();
        }
        else
        {
            std::nth_element(batchList.begin(), batchList.begin() + reinsertEachStep, batchList.end());
        }

        for (int i = 0; i < reinsertEachStep; i++)
        {
            finalOutput.push_back(batchList[i].second);
            leftOutput.erase(batchList[i].second);
            graphutil->recoverAddNode(backupCompletedAdjListGraph, currentAllVex, currentAdjListGraph, batchList[i].second, disjoint_Set);
        }

    }

    std::reverse(finalOutput.begin(), finalOutput.end());

    return finalOutput;
}


double Utils::getRobustness(std::shared_ptr<Graph> graph, std::vector<int> solution)
{
    assert(graph);
    MaxWccSzList.clear();
    std::vector<std::vector<int>> backupCompletedAdjListGraph = graph->adj_list;
    std::vector<std::vector<int>> current_adj_list;
    std::shared_ptr<GraphUtil> graphutil =std::shared_ptr<GraphUtil>(new GraphUtil());
    Disjoint_Set disjoint_Set =  Disjoint_Set(graph->num_nodes);
    std::vector<bool> backupAllVex(graph->num_nodes, false);
    double totalMaxNum = 0.0;
    double temp = 0.0;
    double norm = (double)graph->num_nodes * (double)(graph->num_nodes-1) / 2.0;
    //printf("Norm:%.8f\n", norm);
    for (std::vector<int>::reverse_iterator it = solution.rbegin(); it != solution.rend(); ++it)
    {
        int Node =(*it);
        graphutil->recoverAddNode(backupCompletedAdjListGraph, backupAllVex, current_adj_list, Node, disjoint_Set);
        // calculate the remaining components and its nodes inside
        //std::set<int> lccIDs;
        //for(int i =0;i< graph->num_nodes; i++){
        //    lccIDs.insert(disjoint_Set.unionSet[i]);
        //}
        //double CCDScore = 0.0;
        //for(std::set<int>::iterator it=lccIDs.begin(); it!=lccIDs.end(); it++)
       // {
        //    double num_nodes = (double) disjoint_Set.getRank(*it);
        //    CCDScore += (double) num_nodes * (num_nodes-1) / 2;
       // }

        totalMaxNum += disjoint_Set.CCDScore / norm;
        MaxWccSzList.push_back(disjoint_Set.CCDScore / norm);
        temp = disjoint_Set.CCDScore / norm;
    }

    totalMaxNum = totalMaxNum - temp;
    std::reverse(MaxWccSzList.begin(), MaxWccSzList.end());

    return (double)totalMaxNum / (double)graph->num_nodes;
}


int Utils::getMxWccSz(std::shared_ptr<Graph> graph)
{
    assert(graph);
    Disjoint_Set disjoint_Set =  Disjoint_Set(graph->num_nodes);
    for (int i = 0; i < (int)graph->adj_list.size(); i++)
    {
        for (int j = 0; j < (int)graph->adj_list[i].size(); j++)
        {
            disjoint_Set.merge(i, graph->adj_list[i][j]);
        }
    }
    return disjoint_Set.maxRankCount;
}


std::vector<double> Utils::Betweenness(std::shared_ptr<Graph> _g) {

	int i, j, u, v;
	int Long_max = 4294967295;
	int nvertices = _g->num_nodes;	// The number of vertices in the network
	std::vector<double> CB;
    double norm=(double)(nvertices-1)*(double)(nvertices-2);

	CB.resize(nvertices);

	std::vector<int> d;								// A vector storing shortest distance estimates
	std::vector<int> sigma;							// sigma is the number of shortest paths
	std::vector<double> delta;							// A vector storing dependency of the source vertex on all other vertices
	std::vector< std::vector <int> > PredList;			// A list of predecessors of all vertices

	std::queue <int> Q;								// A priority queue soring vertices
	std::stack <int> S;								// A stack containing vertices in the order found by Dijkstra's Algorithm

	// Set the start time of Brandes' Algorithm

	// Compute Betweenness Centrality for every vertex i
	for (i=0; i < nvertices; i++) {
		/* Initialize */
		PredList.assign(nvertices, std::vector <int> (0, 0));
		d.assign(nvertices, Long_max);
		d[i] = 0;
		sigma.assign(nvertices, 0);
		sigma[i] = 1;
		delta.assign(nvertices, 0);
		Q.push(i);

		// Use Breadth First Search algorithm
		while (!Q.empty()) {
			// Get the next element in the queue
			u = Q.front();
			Q.pop();
			// Push u onto the stack S. Needed later for betweenness computation
			S.push(u);
			// Iterate over all the neighbors of u
			for (j=0; j < (int) _g->adj_list[u].size(); j++) {
				// Get the neighbor v of vertex u
				// v = (ui64) network->vertex[u].edge[j].target;
				v = (int) _g->adj_list[u][j];

				/* Relax and Count */
				if (d[v] == Long_max) {
					 d[v] = d[u] + 1;
					 Q.push(v);
				}
				if (d[v] == d[u] + 1) {
					sigma[v] += sigma[u];
					PredList[v].push_back(u);
				}
			} // End For

		} // End While

		/* Accumulation */
		while (!S.empty()) {
			u = S.top();
			S.pop();
			for (j=0; j < (int)PredList[u].size(); j++) {
				delta[PredList[u][j]] += ((double) sigma[PredList[u][j]]/sigma[u]) * (1+delta[u]);
			}
			if (u != i)
				CB[u] += delta[u];
		}

		// Clear data for the next run
		PredList.clear();
		d.clear();
		sigma.clear();
		delta.clear();
	} // End For

	// End time after Brandes' algorithm and the time difference

    for(int i =0; i<nvertices;++i){
        CB[i]=CB[i]/norm;
    }

	return CB;

} // End of BrandesAlgorithm_Unweighted


