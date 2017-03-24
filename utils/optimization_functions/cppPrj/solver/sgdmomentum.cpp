#include "sgdmomentum.h"

template<typename T>
SGDMomentum<T>::SGDMomentum(const map<string, float> &config):BaseOptimizer<T>(config){

    // try to find key "learning_rate" returns iterator if found
    auto search = BaseOptimizer<T>::m_config.find("learning_rate");

    // if we found "learning_rate" deref. the iterator to get the associated value
    if (search != BaseOptimizer<T>::m_config.end()){
        m_base_lr = search->second;
    }
    else{
        throw("No learning rate set");
    }
}

template<typename T>
Tensor<T> SGDMomentum<T>::Optimize(const Tensor<T> &params, const Tensor<T> &grad_params, const OptimizerState<T> &state){

    //NEEDS FINISHING
    Tensor<T> A(vector<int>({1,2}),{0,0});    
    return A;
}

/*
 * Explicit declare template versions to avoid linker error. (This is needed if we use templates on .cpp files)
*/
template class SGDMomentum<float>;
template class SGDMomentum<double>;
template class SGDMomentum<int>;
