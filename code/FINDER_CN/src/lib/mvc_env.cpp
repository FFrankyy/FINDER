#include "mvc_env.h"
#include "graph.h"
#include <cassert>
#include <random>
#include <algorithm>
#include <set>
#include <queue>
#include <stack>


MvcEnv::MvcEnv(double _norm)
{
norm = _norm;
graph = nullptr;
numCoveredEdges = 0;
CcNum = 1.0;
state_seq.clear();
act_seq.clear();
action_list.clear();
reward_seq.clear();
sum_rewards.clear();
covered_set.clear();
avail_list.clear();
}

MvcEnv::~MvcEnv()
{
    norm = 0;
    graph = nullptr;
    numCoveredEdges = 0;
    state_seq.clear();
    act_seq.clear();
    action_list.clear();
    reward_seq.clear();
    sum_rewards.clear();
    covered_set.clear();
    avail_list.clear();
}

void MvcEnv::s0(std::shared_ptr<Graph> _g)
{
    graph = _g;
    covered_set.clear();
    action_list.clear();
    numCoveredEdges = 0;
    CcNum = 1.0;
    state_seq.clear();
    act_seq.clear();
    reward_seq.clear();
    sum_rewards.clear();
}

double MvcEnv::step(int a)

{
    assert(graph);
    assert(covered_set.count(a) == 0);
    state_seq.push_back(action_list);
    act_seq.push_back(a);
    covered_set.insert(a);
    action_list.push_back(a);

//    double oldCcNum = CcNum;
//    CcNum = getNumofConnectedComponents();

    for (auto neigh : graph->adj_list[a])
        if (covered_set.count(neigh) == 0)
            numCoveredEdges++;
    
//    double r_t = getReward(oldCcNum);
    double r_t = getReward();
    reward_seq.push_back(r_t);
    sum_rewards.push_back(r_t);  

    return r_t;
}


void MvcEnv::stepWithoutReward(int a)

{
    assert(graph);
    assert(covered_set.count(a) == 0);
    covered_set.insert(a);
    action_list.push_back(a);
    for (auto neigh : graph->adj_list[a])
        if (covered_set.count(neigh) == 0)
            numCoveredEdges++;
}


// random
int MvcEnv::randomAction()
{
    assert(graph);
    avail_list.clear();

    for (int i = 0; i < graph->num_nodes; ++i)
        if (covered_set.count(i) == 0)
        {
            bool useful = false;
            for (auto neigh : graph->adj_list[i])
                if (covered_set.count(neigh) == 0)
                {
                    useful = true;
                    break;
                }
            if (useful)
                avail_list.push_back(i);
        }

    assert(avail_list.size());
    int idx = rand() % avail_list.size();
    return avail_list[idx];
}


int MvcEnv::betweenAction()
{
    assert(graph);

    std::map<int,int> id2node;
    std::map<int,int> node2id;

    std::map <int,std::vector<int>> adj_dic_origin;
    std::vector<std::vector<int>> adj_list_reID;


    for (int i = 0; i < graph->num_nodes; ++i)
    {
        if (covered_set.count(i) == 0)
        {
            for (auto neigh : graph->adj_list[i])
            {
                if (covered_set.count(neigh) == 0)
                {
                   if(adj_dic_origin.find(i) != adj_dic_origin.end())
                   {
                       adj_dic_origin[i].push_back(neigh);
                   }
                   else{
                       std::vector<int> neigh_list;
                       neigh_list.push_back(neigh);
                       adj_dic_origin.insert(std::make_pair(i,neigh_list));
                   }
                }
            }
        }

    }


     std::map<int, std::vector<int>>::iterator iter;
     iter = adj_dic_origin.begin();

     int numrealnodes = 0;
     while(iter != adj_dic_origin.end())
     {
        id2node[numrealnodes] = iter->first;
        node2id[iter->first] = numrealnodes;
        numrealnodes += 1;
        iter++;
     }

     adj_list_reID.resize(adj_dic_origin.size());

     iter = adj_dic_origin.begin();
     while(iter != adj_dic_origin.end())
     {
        for(int i=0;i<(int)iter->second.size();++i){
            adj_list_reID[node2id[iter->first]].push_back(node2id[iter->second[i]]);
        }
        iter++;
     }


    std::vector<double> BC = Betweenness(adj_list_reID);
    std::vector<double>::iterator biggest_BC = std::max_element(std::begin(BC), std::end(BC));
    int maxID = std::distance(std::begin(BC), biggest_BC);
    int idx = id2node[maxID];
//    printGraph();
//    printf("\n maxBetID:%d, value:%.6f\n",idx,BC[maxID]);
    return idx;
}

bool MvcEnv::isTerminal()
{
    assert(graph);
//    printf ("num edgeds:%d\n", graph->num_edges);
//    printf ("numCoveredEdges:%d\n", numCoveredEdges);
    return graph->num_edges == numCoveredEdges;
}


double MvcEnv::getReward()
{
    double orig_node_num = (double) graph->num_nodes;
    return -(double)getRemainingCNDScore()/(orig_node_num*orig_node_num*(orig_node_num-1)/2);
}

//double MvcEnv::getReward(double oldCcNum)
//{
//    return (CcNum - oldCcNum) / CcNum*graph->num_nodes ;
//
//}

void MvcEnv::printGraph()
{
    printf("edge_list:\n");
    printf("[");
    for (int i = 0; i < (int)graph->edge_list.size();i++)
    {
    printf("[%d,%d],",graph->edge_list[i].first,graph->edge_list[i].second);
    }
    printf("]\n");


    printf("covered_set:\n");

    std::set<int>::iterator it;
    printf("[");
    for (it=covered_set.begin();it!=covered_set.end();it++)
    {
        printf("%d,",*it);
    }
    printf("]\n");

}



double MvcEnv::getRemainingCNDScore()
{
    assert(graph);
    Disjoint_Set disjoint_Set =  Disjoint_Set(graph->num_nodes);

    for (int i = 0; i < graph->num_nodes; i++)
    {
        if (covered_set.count(i) == 0)
        {
            for (auto neigh : graph->adj_list[i])
            {
                if (covered_set.count(neigh) == 0)
                {
                    disjoint_Set.merge(i, neigh);
                }
            }
        }
    }

    std::set<int> lccIDs;
    for(int i =0;i< graph->num_nodes; i++){
        lccIDs.insert(disjoint_Set.unionSet[i]);
    }

    double CCDScore = 0.0;
    for(std::set<int>::iterator it=lccIDs.begin(); it!=lccIDs.end(); it++)
    {
        double num_nodes = (double) disjoint_Set.getRank(*it);
        CCDScore += (double) num_nodes * (num_nodes-1) / 2;
    }

    return CCDScore;
}



double MvcEnv::getMaxConnectedNodesNum()
{
    assert(graph);
    Disjoint_Set disjoint_Set =  Disjoint_Set(graph->num_nodes);

    for (int i = 0; i < graph->num_nodes; i++)
    {
        if (covered_set.count(i) == 0)
        {
            for (auto neigh : graph->adj_list[i])
            {
                if (covered_set.count(neigh) == 0)
                {
                    disjoint_Set.merge(i, neigh);
                }
            }
        }
    }
    return (double)disjoint_Set.maxRankCount;
}


std::vector<double> MvcEnv::Betweenness(std::vector< std::vector <int> > adj_list) {

	int i, j, u, v;
	int Long_max = 4294967295;
	int nvertices = adj_list.size();	// The number of vertices in the network
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
			for (j=0; j < (int) adj_list[u].size(); j++) {
				// Get the neighbor v of vertex u
				// v = (ui64) network->vertex[u].edge[j].target;
				v = (int) adj_list[u][j];

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
        if (norm == 0)
        {
            CB[i] = 0;
        }
        else
        {
            CB[i]=CB[i]/norm;
        }
    }

	return CB;

} // End of BrandesAlgorithm_Unweighted