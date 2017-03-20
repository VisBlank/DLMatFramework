#include "adam.h"

template <typename T>
Adam<T>::Adam()
{

}

template<typename T>
tuple<Tensor<T>, Tensor<T> > Adam<T>::Optimize(const Tensor<T> &params, const Tensor<T> &grad_params, const Tensor<T> &state){
    cout << "Adam optimizer" << endl;
    Tensor<T> A(vector<int>({1,2}),{0,0});
    Tensor<T> B(vector<int>({1,2}),{0,0});
    return make_tuple(A,B);
}

/*
 * Explicit declare template versions to avoid linker error. (This is needed if we use templates on .cpp files)
*/
template class Adam<float>;
template class Adam<double>;
template class Adam<int>;
