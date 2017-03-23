#ifndef DEEPLEARNINGMODEL_H
#define DEEPLEARNINGMODEL_H

#include <tuple>
#include <iostream>
#include <string>

#include "layers/layercontainer.h"
#include "loss/baseloss.h"
#include "utils/reverse_range_based.h"

using namespace std;

class DeepLearningModel
{
private:
    LayerContainer m_layers;
    shared_ptr<BaseLoss> m_loss;
    bool m_isTraining;
public:
    // Delete default constructor
    DeepLearningModel() = delete;

    DeepLearningModel(const LayerContainer &lc, shared_ptr<BaseLoss> bl):m_layers(lc){
        m_loss = bl;
        m_isTraining = false;
        InitWeights();
    }

    Tensor<float> Predict(const Tensor<float> &input);
    float Loss(const Tensor<float> &X, const Tensor<float> &Y);
    bool IsTraining() const {return m_isTraining;}
    void IsTraining(const bool &flag) {m_isTraining = flag;}
    LayerContainer &GetLayers() { return ref(m_layers);}

private:
    void InitWeights();
};

#endif // DEEPLEARNINGMODEL_H
