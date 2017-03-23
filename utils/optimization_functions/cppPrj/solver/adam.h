#ifndef ADAM_H
#define ADAM_H
#include "baseoptimizer.h"

template <typename T>
class Adam : public BaseOptimizer<T>
{
public:
    Adam();
    Tensor<T> Optimize(const Tensor<T> &params, const Tensor<T> &grad_params, const OptimizerState<T> &state) override;
};

#endif // ADAM_H
