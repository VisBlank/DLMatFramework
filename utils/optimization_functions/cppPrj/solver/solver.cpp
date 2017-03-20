#include "solver.h"

template <typename T>
Solver<T>::Solver(DeepLearningModel &net, const OptimizerType &type, const map<string, float> &config):m_net(net){
    /*switch (type) {
    case OptimizerType::SGD:
        m_optimizer = new SGD();
        break;
    case OptimizerType::SGDMomentum:
        m_optimizer = unique_ptr<BaseOptimizer>(new SGDMomentum());
        break;
    case OptimizerType::Adam:
        m_optimizer = unique_ptr<BaseOptimizer>(new Adam());
        break;
    default:
        throw invalid_argument("Optimizer not implemented.");
        break;
    }*/
    m_optimizer = new Adam<T>();
}

template <typename T>
void Solver<T>::Train(){

}

template <typename T>
void Solver<T>::SetEpochs(int epochs){
    m_epochs = epochs;
}

template <typename T>
void Solver<T>::SetBatchSize(int batch){
    m_batchSize = batch;
}

/*
 * Explicit declare template versions to avoid linker error. (This is needed if we use templates on .cpp files)
*/
template class Solver<float>;
template class Solver<double>;
template class Solver<int>;
