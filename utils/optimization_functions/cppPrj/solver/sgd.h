#ifndef SGD_H
#define SGD_H
#include "baseoptimizer.h"
using namespace std;


template <typename T>
class SGD : public BaseOptimizer<T>
{
public:
    // Delete the default constructor so it will force everyone to use SGD passing a config
    SGD() = delete;
    SGD(const map<string,float> &config);
    Tensor<T> Optimize(const Tensor<T> &params, const Tensor<T> &grad_params, const OptimizerState<T> &state) override;
    ~SGD(){
        cout << "SGD destructor" << endl;
    }
};

#endif // SGD_H
