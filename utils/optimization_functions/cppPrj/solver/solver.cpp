#include "solver.h"

Solver::Solver(DeepLearningModel &net, const OptimizerType &type, const map<string, float> &config):m_net(net){
    switch (type) {
    case OptimizerType::T_SGD:
        m_optimizer = unique_ptr<BaseOptimizer<float>>(new SGD<float>());
        break;
    case OptimizerType::T_SGD_Momentum:
        m_optimizer = unique_ptr<BaseOptimizer<float>>(new SGDMomentum<float>());
        break;
    case OptimizerType::T_Adam:
        m_optimizer = unique_ptr<BaseOptimizer<float>>(new Adam<float>());
        break;
    default:
        throw invalid_argument("Optimizer not implemented.");
        break;
    }    
}

void Solver::Train(){
    Step();
}

void Solver::SetEpochs(int epochs){
    m_epochs = epochs;
}

void Solver::SetBatchSize(int batch){
    m_batchSize = batch;
}

void Solver::Step(){
    cout << "Solver::Step" << endl;        
    Tensor<float> A(vector<int>({1,2}),{0,0});
    Tensor<float> B(vector<int>({1,2}),{0,0});
    Tensor<float> C(vector<int>({1,2}),{0,0});
    m_optimizer->Optimize(A, B, C);
}

/*
 * Explicit declare template versions to avoid linker error. (This is needed if we use templates on .cpp files)
*/
//template class Solver<float>;
//template class Solver<double>;
//template class Solver<int>;
