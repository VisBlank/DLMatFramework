#ifndef BASEOPTIMIZER_H
#define BASEOPTIMIZER_H

#include <tuple>
#include "utils/tensor.h"

using namespace std;

template <typename T>
class BaseOptimizer
{
public:    
    virtual tuple<Tensor<T>, Tensor<T>> Optimize(const Tensor<T> &params, const Tensor<T> &grad_params, const Tensor<T> &state) = 0;
    virtual ~BaseOptimizer(){
        cout << "BaseOptimizer destructor" << endl;
    }
};

#endif // BASEOPTIMIZER_H
