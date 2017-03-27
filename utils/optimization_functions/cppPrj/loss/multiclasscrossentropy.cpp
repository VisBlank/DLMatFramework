#include "multiclasscrossentropy.h"

MultiClassCrossEntropy::MultiClassCrossEntropy()
{

}

tuple<float, Tensor<float> > MultiClassCrossEntropy::GetLossAndGradients(const Tensor<float> &scores, const Tensor<float> &targets){
    auto N = scores.GetRows();
    /*
     * Considering that the socres are already converted to probabilities (After softmax activation)
     * Get the indexes of the correct classes
     * */


    cout << "From MultiClassCrossEntropy" << endl;
    return make_tuple(0.1F,Tensor<float>());
}
