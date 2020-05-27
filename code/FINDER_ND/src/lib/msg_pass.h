#ifndef MSG_PASS_H
#define MSG_PASS_H

#include "graph_struct.h"
#include <memory>
#include <vector>

std::shared_ptr<sparseMatrix> n2n_construct(GraphStruct* graph);

std::shared_ptr<sparseMatrix> e2n_construct(GraphStruct* graph);

std::shared_ptr<sparseMatrix> n2e_construct(GraphStruct* graph);

std::shared_ptr<sparseMatrix> e2e_construct(GraphStruct* graph);

std::shared_ptr<sparseMatrix> subg_construct(GraphStruct* graph);

#endif