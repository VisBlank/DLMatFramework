#include "adam.h"

template<typename T>
Adam<T>::Adam(const map<string, float> &config):BaseOptimizer<T>(config){

}

template<typename T>
Tensor<T> Adam<T>::Optimize(const Tensor<T> &params, const Tensor<T> &grad_params, const OptimizerState<T> &state){
    cout << "Adam optimizer" << endl;
    Tensor<T> A(vector<int>({1,2}),{0,0});    
    return A;
}

/*
 * Explicit declare template versions to avoid linker error. (This is needed if we use templates on .cpp files)
*/
template class Adam<float>;
template class Adam<double>;
template class Adam<int>;
