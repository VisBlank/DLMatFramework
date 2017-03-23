#ifndef BASEOPTIMIZER_H
#define BASEOPTIMIZER_H

#include <tuple>
#include <map>
#include "utils/tensor.h"

using namespace std;

template <typename T>
class OptimizerState {
public:
    // State vectors used on SGD with momentum
    Tensor<T> velocity;

    // State vectors used on Adam
    Tensor<T> time;
    Tensor<T> m;
    Tensor<T> v;
};

template <typename T>
class BaseOptimizer
{
public:    
    virtual Tensor<T> Optimize(const Tensor<T> &params, const Tensor<T> &grad_params, const OptimizerState<T> &state) = 0;    
    virtual ~BaseOptimizer(){
        cout << "BaseOptimizer destructor" << endl;
    }
};

#endif // BASEOPTIMIZER_H
