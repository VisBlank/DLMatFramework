#include "layercontainer.h"

LayerContainer::LayerContainer(){

}

typename list<string>::iterator LayerContainer::begin(){
    return m_layerNamesList.begin();
}

typename list<string>::iterator LayerContainer::end() {
    return m_layerNamesList.end();
}

typename list<string>::const_iterator LayerContainer::begin() const {
    return m_layerNamesList.begin();
}

typename list<string>::const_iterator LayerContainer::end() const {
    return m_layerNamesList.end();
}

void LayerContainer::operator<=(const LayerMetaData &layerData){
    shared_ptr<BaseLayer> layer;
    switch (layerData.type) {
    case TInput:
        layer = shared_ptr<BaseLayer>(new InputLayer(layerData.name, layerData.p1, layerData.p2, layerData.p3, layerData.p4));
        break;
    case TRelu:
        layer = shared_ptr<BaseLayer>(new Relu(layerData.name, m_currentLayer));
        break;
    case TSigmoid:
        layer = shared_ptr<BaseLayer>(new Sigmoid(layerData.name, m_currentLayer));
        break;
    case TFullyConnected:
        layer = shared_ptr<BaseLayer>(new FullyConnected(layerData.name, m_currentLayer,layerData.p1));
        break;
    case TSoftMax:
        layer = shared_ptr<BaseLayer>(new SoftMax(layerData.name, m_currentLayer));
        break;
    default:
        throw invalid_argument("Layer not implemented.");
        break;
    }
    m_hashMapLayers.insert(make_pair(layerData.name,shared_ptr<BaseLayer>(layer)));
    m_layerNamesList.push_back(layerData.name);
    m_numLayers++;
    // Non-thread safe way to hold the last inserted layer
    m_currentLayer = layer;
}

shared_ptr<BaseLayer> LayerContainer::operator()(const string &layerName){
    return m_hashMapLayers.at(layerName);
}

int LayerContainer::GetNumLayers() const {
    return m_numLayers;
}

vector<list<shared_ptr<BaseLayer> > > LayerContainer::GetGraph() const {
    return m_adjacency_vector;
}

void LayerContainer::BuildGraph(){
    // Populate the Adjacency vector
}
