#ifndef INPUTLAYER_H
#define INPUTLAYER_H
#include "baselayer.h"
using namespace std;

class InputLayer : public BaseLayer
{
public:
    InputLayer(const string &name, int numRows, int numCols, int numChannels, int batchSize);

    Tensor<float> ForwardPropagation(const Tensor<float> &input) override;
    LayerGradient<float> BackwardPropagation(const Tensor<float> &dout) override;
};
#endif // INPUTLAYER_H
