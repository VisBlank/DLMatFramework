#include "batchnorm.h"

BatchNorm::BatchNorm(const string &name, shared_ptr<BaseLayer> inLayer, float eps, float momentum){
    m_name = name;
    m_inputLayer = inLayer;
    m_isTraining = true;
    m_eps = eps;
    m_momentum = momentum;

    // BatchNorm does not change the shape of it's input
    if (m_inputLayer != nullptr){
        m_activationShape = m_inputLayer->GetActivationShape();

        // Initialize running_mean and running_var as zeros
        auto D = m_activationShape[1];
        m_running_mean = MathHelper<float>::Zeros(vector<int>{1,D});
        m_running_var = MathHelper<float>::Zeros(vector<int>{1,D});

        // Initialize weights and bias (TODO....)
        m_weights = MathHelper<float>::Ones(vector<int>({1,D}));
        m_bias = MathHelper<float>::Zeros(vector<int>({1,D}));
    }
}

Tensor<float> BatchNorm::ForwardPropagation(const Tensor<float> &input){
    Tensor<float> activation;
    //auto N = input.GetBatch();
    // For now batch are the same as rows (TODO: Change this...)
    auto N = (float)input.GetRows();

    if (m_isTraining){
        // Step 1: Calculate the mean of the batch
        auto mu = MathHelper<float>::Sum(input,0)/N;

        // Step 2: Subtract the mean from each collumn
        m_xmu = input - mu.Repmat(N,1);

        // Step 3: Calculate the denominator
        auto sq = m_xmu.EltWisePow(2);

        // Step 4: Calculate the variance
        m_var = MathHelper<float>::Sum(sq,0)/N;

        // Step 5: Add eps for numerical stability then do sqrt
        m_sqrtvar = MathHelper<float>::Sqrt(m_var + m_eps);

        // Setp 6: Invert the Square root
        auto ivar = 1.0/m_sqrtvar;

        // Step 7: Do Normalization
        m_xhat = m_xmu.EltWiseMult(ivar.Repmat(N,1));

        // Step 8: Nor the two transformations steps
        auto gammax = (m_weights.Repmat(N,1)).EltWiseMult(m_xhat);

        // Step 9: Adjust with bias (Batchnorm output)
        activation = gammax + m_bias.Repmat(N,1);

        // Calculate the running mean and variance to be used latter on prediction
        m_running_mean = (m_running_mean * m_momentum) + (mu * (1.0 - m_momentum));
        m_running_var = (m_running_var * m_momentum) + (m_var * (1.0 - m_momentum));

    } else {
        // Normalization with calculated mean and variance
        auto runvar_sqrt = MathHelper<float>::Sqrt(m_running_var+m_eps);
        auto xbar = (input - m_running_mean.Repmat(N,1)).EltWiseDiv(runvar_sqrt.Repmat(N,1));

        // Apply the learned gamma(weights) and betas(bias) during training time
        activation = (m_weights.Repmat(N,1)).EltWiseMult(xbar) + m_bias.Repmat(N,1);
    }
    // Cache information for backpropagation
    m_previousInput = input;
    m_activation = activation;

    return activation;
}

LayerGradient<float> BatchNorm::BackwardPropagation(const LayerGradient<float> &dout){

}
