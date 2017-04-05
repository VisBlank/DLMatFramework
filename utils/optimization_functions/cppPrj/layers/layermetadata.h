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
    LayerMetaData(const string &pName, LayerType pType, int pP1){
        name = pName;
        type = pType;
        p1 = pP1;
    }

    LayerMetaData(const string &pName, LayerType pType){
        name = pName;
        type = pType;
    }

    LayerMetaData(const string &pName, LayerType pType, int pP1, int pP2, int pP3, int pP4){
        name = pName;
        type = pType;
        p1 = pP1;
        p2 = pP2;
        p3 = pP3;
        p4 = pP4;
    }

    LayerMetaData(const string &pName, LayerType pType, float pP1){
        name = pName;
        type = pType;
        pf1 = pP1;
    }

    LayerMetaData(const string &pName, LayerType pType, float pP1, float pP2){
        name = pName;
        type = pType;
        pf1 = pP1;
        pf2 = pP2;
    }

    // Integer parameters
    int GetP1() const {return p1;}
    int GetP2() const {return p2;}
    int GetP3() const {return p3;}
    int GetP4() const {return p4;}

    // Float parameters
    float GetPF1() const {return pf1;}
    float GetPF2() const {return pf2;}
    float GetPF3() const {return pf3;}
    float GetPF4() const {return pf4;}

    // Layer name/type
    LayerType GetType() const {return type;}
    string GetName() const {return name;}
private:
    string name;
    LayerType type;
    int p1;
    int p2;
    int p3;
    int p4;
    float pf1;
    float pf2;
    float pf3;
    float pf4;
};

#endif // LAYERMETADATA_H
