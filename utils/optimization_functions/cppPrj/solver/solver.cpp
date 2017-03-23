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
    // Indicate to model that training phase started
    m_net.IsTraining(true);

    int num_train = m_data.GetTrainSize();
    int iterations_per_epoch = ceil((float)num_train / (float)m_batchSize);
    int num_iterations = iterations_per_epoch * m_epochs;    
    cout << "Epochs: " << m_epochs << " Iterations/Epoch: " << iterations_per_epoch << endl;

    // Indicate the dataset class that we want to auto-shuffle every iterations_per_epoch iterations
    m_data.ShuffleEveryNIterations(iterations_per_epoch);

    for (int t=0; t < num_iterations; ++t){
        // Do a solver step (Get loss/gradients, update weights(SGD, Adam, etc...)
        Step();
    }

    // Indicate to model that training phase is over
    m_net.IsTraining(false);
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

    // Get model loss and gradients (The model Loss method will invoke the backpropagation)
    auto LossGrad = m_net.Loss(batch.X, batch.Y);
    auto grad = get<1>(LossGrad);
    m_loss_history.push_back(get<0>(LossGrad));

    // Perform parameter update on each layer (Not that not all layer has parameters)
    for (auto &layerName :m_net.GetLayers()){
        // Get the layer instance
        auto layer = m_net.GetLayers()(layerName);
        // Continue if layer has no parameter
        if (!layer->HasParameter()) continue;

        // Select the weight and bias references (Those will be changed by the Optimizers)
        auto weights = layer->GetWeightsRef();
        auto bias = layer->GetBiasRef();

        // Add regularization (TODO)

        // Use optimizers to calculate new weights and biases (TODO: Handle Optimizer state as a class)
        OptimizerState<float> opt;
        auto newWeights = m_optimizer->Optimize(weights, layer->GetGradientRef().dWeights, opt);
        auto newBias = m_optimizer->Optimize(bias, layer->GetGradientRef().dBias, opt);

        // Update weights and biases on the model
        weights = newWeights;
        bias = newBias;

    }
}

/*
 * Explicit declare template versions to avoid linker error. (This is needed if we use templates on .cpp files)
*/
//template class Solver<float>;
//template class Solver<double>;
//template class Solver<int>;
