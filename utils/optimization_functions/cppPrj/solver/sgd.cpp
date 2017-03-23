#include "sgd.h"
#include <map>
template <typename T>
SGD<T>::SGD(const map<string, float> &config)
{
    m_config = config;
    m_base_lr = m_config["learning_rate"];
}

template<typename T>
Tensor<T> SGD<T>::Optimize(const Tensor<T> &params, const Tensor<T> &grad_params, const OptimizerState<T> &state){
    //cout << "SGD optimizer" << endl;

    // Gradient descent approach (Simpliest optimizer, no use of states)
    auto weights = params - (grad_params*m_base_lr);

    return weights;
}

/*
 * Explicit declare template versions to avoid linker error. (This is needed if we use templates on .cpp files)
*/
template class SGD<float>;
template class SGD<double>;
template class SGD<int>;

