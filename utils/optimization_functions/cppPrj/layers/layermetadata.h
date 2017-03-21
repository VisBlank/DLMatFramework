/*
    Pojo for Layer MetaData
    http://en.cppreference.com/w/cpp/language/enum
    http://en.cppreference.com/w/cpp/language/aggregate_initialization
*/
#ifndef LAYERMETADATA_H
#define LAYERMETADATA_H

#include <string>
using namespace std;

enum LayerType { TInput, TFullyConnected, TRelu, TSigmoid, TSoftMax, TConvolution, TMaxPooling, TAveragePooling, TElementWiseAdd, TDropout, TBatchNorm, TSpatialBatchNorm };

class LayerMetaData{
public:
    string name;
    LayerType type;
    int p1;
    int p2;
    int p3;
    int p4;
};

#endif // LAYERMETADATA_H
