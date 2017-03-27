#include "multiclasscrossentropy.h"

MultiClassCrossEntropy::MultiClassCrossEntropy()
{

}

tuple<float, Tensor<float> > MultiClassCrossEntropy::GetLossAndGradients(const Tensor<float> &scores, const Tensor<float> &targets){
    auto N = scores.GetRows();
    //Considering that the socres are already converted to probabilities (After softmax activation)

    // Get the indexes of the correct classes
    auto argMaxTargets = MathHelper<float>::MaxTensor(targets,1);
    auto idxCorrect = argMaxTargets.second;

    // Get the probabilities of the correct class
    auto selScoresTargets = scores.EltWiseMult(targets);
    auto probCorrect = MathHelper<float>::GetNonZero(selScoresTargets);

    // Calculate the loss
    auto loss = -MathHelper<float>::SumVec(MathHelper<float>::Log(probCorrect))/N;

    // Get the gradient of the loss w.r.t
    auto gradients = scores;
    auto gradients_correct = probCorrect - 1;

    // Update on each row of gradients (using the index from idxCorrect) the correct gradient (gradients_correct)
    auto rowsGradients = gradients.GetRows();
    auto idxCorrectIt = idxCorrect.begin();
    auto gradients_correct_it = gradients_correct.begin();
    for (auto rows = 0; rows < rowsGradients; ++rows){
        // Select
        gradients(rows, *idxCorrectIt) = *gradients_correct_it;
        gradients_correct_it++;
        idxCorrectIt++;
    }
    // Take the effect of the batch size
    gradients = gradients / N;

    return make_tuple(loss,gradients);
}
