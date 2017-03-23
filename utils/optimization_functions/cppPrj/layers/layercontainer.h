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
#include <map>
#include <unordered_map>
#include <unordered_set>
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
    void operator<=(const LayerMetaData &layerData);
    shared_ptr<BaseLayer> operator()(const string &layerName);
    int GetNumLayers() const;
    vector<list<shared_ptr<BaseLayer>>> GetGraph() const;
    void BuildGraph();

    /*typename unordered_map<string,shared_ptr<BaseLayer>>::iterator begin();
    typename unordered_map<string,shared_ptr<BaseLayer>>::iterator end();
    typename unordered_map<string,shared_ptr<BaseLayer>>::const_iterator begin() const;
    typename unordered_map<string,shared_ptr<BaseLayer>>::const_iterator end() const;*/
    // We want to have the layers with the same order as we inserted them
    typename list<string>::iterator begin();
    typename list<string>::iterator end();
    typename list<string>::const_iterator begin() const;
    typename list<string>::const_iterator end() const;

    typename list<string>::reverse_iterator rbegin();
    typename list<string>::reverse_iterator rend();
    typename list<string>::const_reverse_iterator rbegin() const;
    typename list<string>::const_reverse_iterator rend() const;

private:
    // Hash map of layers unique pointers (Remember that they should be moved here...)
    unordered_map<string,shared_ptr<BaseLayer>> m_hashMapLayers;
    list<string> m_layerNamesList;

    // Number of layers
    int m_numLayers = 0;

    // Used to store the last inserted layer
    shared_ptr<BaseLayer> m_currentLayer = nullptr;

    // Adjacency vector (vector of list of pointers to layers)
    vector<list<shared_ptr<BaseLayer>>> m_adjacency_vector;

};

#endif // LAYERCONTAINER_H
