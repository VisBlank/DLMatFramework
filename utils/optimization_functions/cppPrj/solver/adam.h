#ifndef ADAM_H
#define ADAM_H
#include "baseoptimizer.h"

template <typename T>
class Adam : public BaseOptimizer<T>
{
public:
    // Delete the default constructor so it will force everyone to use SGD passing a config
    Adam() = delete;
    Adam(const map<string,float> &config);
    Tensor<T> Optimize(const Tensor<T> &params, const Tensor<T> &grad_params, const OptimizerState<T> &state) override;
};

#endif // ADAM_H
