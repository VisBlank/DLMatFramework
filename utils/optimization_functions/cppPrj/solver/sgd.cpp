#include "sgd.h"

template <typename T>
SGD<T>::SGD()
{

}

template<typename T>
tuple<Tensor<T>, Tensor<T> > SGD<T>::Optimize(const Tensor<T> &params, const Tensor<T> &grad_params, const Tensor<T> &state){
    cout << "SGD optimizer" << endl;

    Tensor<T> A(vector<int>({1,2}),{0,0});
    Tensor<T> B(vector<int>({1,2}),{0,0});
    return make_tuple(A,B);
}

/*
 * Explicit declare template versions to avoid linker error. (This is needed if we use templates on .cpp files)
*/
template class SGD<float>;
template class SGD<double>;
template class SGD<int>;

