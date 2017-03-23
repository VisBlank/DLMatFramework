#ifndef SOLVER_H
#define SOLVER_H

#include <iostream>
#include <map>
#include <deque>
#include <string>
#include <memory>
#include <cmath>
#include "classifier/deeplearningmodel.h"
#include "baseoptimizer.h"
#include "sgd.h"
#include "sgdmomentum.h"
#include "adam.h"
#include "utils/tensor.h"
#include "utils/dataset.h"

using namespace std;

enum OptimizerType { T_SGD, T_SGD_Momentum, T_Adam};

class Solver
{
private:
    DeepLearningModel &m_net;
    Dataset<float> &m_data;
    int m_epochs = 100;
    int m_batchSize = 1;
    unique_ptr<BaseOptimizer<float>> m_optimizer = nullptr;    
    vector<float> m_loss_history;
    bool m_verbose = true;
    int m_print_every = 1000;
public:
    Solver() = delete;
    Solver(DeepLearningModel &net, Dataset<float> &data, const OptimizerType &type, const map<string,float> &config);
    void Train();
    void SetEpochs(int epochs);
    void SetBatchSize(int batch);
    vector<float> GetLossHistory() const;
private:
    void Step();
};

#endif // SOLVER_H
