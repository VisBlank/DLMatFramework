#include "solver.h"

Solver::Solver(DeepLearningModel &net, Dataset<float> &data, const OptimizerType &type, const map<string, float> &config):m_net(net),m_data(data){
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
    int num_train = m_data.GetTrainSize();
    int iterations_per_epoch = ceil((float)num_train / (float)m_batchSize);
    int num_iterations = iterations_per_epoch * m_epochs;

    for (int t=0; t < num_iterations; ++t){
        Step();
    }
}

void Solver::SetEpochs(int epochs){
    m_epochs = epochs;
}

void Solver::SetBatchSize(int batch){
    m_batchSize = batch;
}

vector<float> Solver::GetLossHistory() const {
     return m_loss_history;
}

void Solver::Step(){    
    // Select a mini-batch
    Batch<float> batch = m_data.GetBatch(m_batchSize);

    // Get model loss and gradients
    auto LossGrad = m_net.Loss(batch.X, batch.Y);
    m_loss_history.push_back(get<0>(LossGrad));

    // Perform parameter update

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
