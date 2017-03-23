#include "sgdmomentum.h"

template<typename T>
SGDMomentum<T>::SGDMomentum(const map<string, float> &config){

}

template<typename T>
Tensor<T> SGDMomentum<T>::Optimize(const Tensor<T> &params, const Tensor<T> &grad_params, const OptimizerState<T> &state){
    cout << "SGDMomentum optimizer" << endl;
    Tensor<T> A(vector<int>({1,2}),{0,0});    
    return A;
}

/*
 * Explicit declare template versions to avoid linker error. (This is needed if we use templates on .cpp files)
*/
template class SGDMomentum<float>;
template class SGDMomentum<double>;
template class SGDMomentum<int>;
