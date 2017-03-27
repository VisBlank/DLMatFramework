#include "multiclasscrossentropy.h"

MultiClassCrossEntropy::MultiClassCrossEntropy()
{

}

tuple<float, Tensor<float> > MultiClassCrossEntropy::GetLossAndGradients(const Tensor<float> &scores, const Tensor<float> &targets){
    auto N = scores.GetRows();
    //Considering that the socres are already converted to probabilities (After softmax activation)

    // Get the indexes of the correct classes
    auto argMaxTargets = MathHelper<float>::MaxTensor(targets,1);

    // Get the probabilities of the correct class
    auto probCorrect = argMaxTargets.second;

    // Calculate the loss
    auto loss = 0.1F;

    // Get the gradient of the loss w.r.t
    auto gradients = scores;
    auto gradients_correct = probCorrect - 1;
    // Put the calculated gradients back on...

    gradients = gradients / N;


    cout << "From MultiClassCrossEntropy" << endl;
    return make_tuple(loss,gradients);
}
