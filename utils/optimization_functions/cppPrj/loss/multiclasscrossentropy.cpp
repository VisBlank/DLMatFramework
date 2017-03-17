#include "multiclasscrossentropy.h"

MultiClassCrossEntropy::MultiClassCrossEntropy()
{

}

tuple<float, Tensor<float> > MultiClassCrossEntropy::GetLossAndGradients(const Tensor<float> &scores, const Tensor<float> &targets)
{
    cout << "From MultiClassCrossEntropy" << endl;
    return make_tuple(0.1F,Tensor<float>());
}
