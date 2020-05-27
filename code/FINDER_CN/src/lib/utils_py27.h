#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <set>
#include <memory>
#include "disjoint_set.h"
#include "graph_utils.h"

class Utils
{
public:
    Utils();
    double  getRobustness(std::vector<int> edges_from, std::vector<int> edges_to,int num_nodes, std::vector<int> solution);
    std::vector<double> MaxWccSzList;
};

#endif