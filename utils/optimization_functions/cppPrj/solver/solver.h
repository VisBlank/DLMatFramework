#ifndef SOLVER_H
#define SOLVER_H

#include <iostream>
#include <map>
#include <string>
#include <memory>
#include "classifier/deeplearningmodel.h"
#include "baseoptimizer.h"
#include "sgd.h"
#include "sgdmomentum.h"
#include "adam.h"

using namespace std;

enum OptimizerType { T_SGD, T_SGD_Momentum, T_Adam};

class Solver
{
private:
    DeepLearningModel &m_net;
    int m_epochs = 100;
    int m_batchSize = 1;
    unique_ptr<BaseOptimizer<float>> m_optimizer = nullptr;
public:
    Solver() = delete;
    Solver(DeepLearningModel &net, const OptimizerType &type, const map<string,float> &config);
    void Train();
    void SetEpochs(int epochs);
    void SetBatchSize(int batch);
private:
    void Step();
};

#endif // SOLVER_H
