#ifndef DEEPLEARNINGMODEL_H
#define DEEPLEARNINGMODEL_H
#include "layers/layercontainer.h"
#include "loss/baseloss.h"

class DeepLearningModel
{
private:
    LayerContainer m_layers;
    shared_ptr<BaseLoss> m_loss;
public:
    // Delete default constructor
    DeepLearningModel() = delete;

    DeepLearningModel(const LayerContainer &lc, shared_ptr<BaseLoss> bl):m_layers(lc){
        m_loss = bl;
        Tensor<float> A(vector<int>({4,1}),{0,1,1,0});
        Tensor<float> B(vector<int>({4,1}),{0,1,1,0});
        auto loss = m_loss->GetLossAndGradients(A,B);
    }
};

#endif // DEEPLEARNINGMODEL_H
