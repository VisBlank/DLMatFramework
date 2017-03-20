/*
    Layer container
    References:
    http://thispointer.com/unordered_map-usage-tutorial-and-example/
    http://codereview.stackexchange.com/questions/114304/c-graph-implementation
    http://blog.coldflake.com/posts/Testing-C++-with-a-new-Catch/
*/
#ifndef LAYERCONTAINER_H
#define LAYERCONTAINER_H
#include "baselayer.h"
#include "layermetadata.h"
#include <string>
#include <unordered_map>
#include <algorithm>
#include <utility>
#include <memory>
#include <vector>
#include <list>

#include <layers/inputlayer.h>
#include <layers/relu.h>
#include <layers/sigmoid.h>
#include <layers/fullyconnected.h>
#include <layers/softmax.h>

using namespace std;

class LayerContainer
{
public:
    LayerContainer();

    void operator<=(const LayerMetaData &layerData){
        shared_ptr<BaseLayer> layer;
        switch (layerData.type) {
        case TInput:
            layer = shared_ptr<BaseLayer>(new InputLayer(layerData.name));
            break;
        case TRelu:
            layer = shared_ptr<BaseLayer>(new Relu(layerData.name, m_currentLayer));
            break;
        case TSigmoid:
            layer = shared_ptr<BaseLayer>(new Sigmoid(layerData.name, m_currentLayer));
            break;
        case TFullyConnected:
            layer = shared_ptr<BaseLayer>(new FullyConnected(layerData.name, m_currentLayer));
            break;
        case TSoftMax:
            layer = shared_ptr<BaseLayer>(new SoftMax(layerData.name, m_currentLayer));
            break;
        default:
            throw invalid_argument("Layer not implemented.");
            break;
        }
        m_hashMapLayers.insert(make_pair(layerData.name,shared_ptr<BaseLayer>(layer)));
        m_numLayers++;
        // Non-thread safe way to hold the last inserted layer
        m_currentLayer = layer;
    }

    int GetNumLayers() const { return m_numLayers;}
    vector<list<shared_ptr<BaseLayer>>> GetGraph() const {return m_adjacency_vector;}

    void BuildGraph(){
        // Populate the Adjacency vector
    }

private:
    // Hash map of layers unique pointers (Remember that they should be moved here...)
    unordered_map<string,shared_ptr<BaseLayer>> m_hashMapLayers;

    // Number of layers
    int m_numLayers = 0;

    // Used to store the last inserted layer
    shared_ptr<BaseLayer> m_currentLayer = nullptr;

    // Adjacency vector (vector of list of pointers to layers)
    vector<list<shared_ptr<BaseLayer>>> m_adjacency_vector;

};

#endif // LAYERCONTAINER_H
