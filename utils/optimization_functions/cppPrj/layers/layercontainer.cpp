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

typename list<string>::reverse_iterator LayerContainer::rbegin(){
    return m_layerNamesList.rbegin();
}
typename list<string>::reverse_iterator LayerContainer::rend(){
    return m_layerNamesList.rend();
}
typename list<string>::const_reverse_iterator LayerContainer::rbegin() const {
    return m_layerNamesList.rbegin();
}
typename list<string>::const_reverse_iterator LayerContainer::rend() const {
    return m_layerNamesList.rend();
}

void LayerContainer::operator<=(const LayerMetaData &layerData){
    shared_ptr<BaseLayer> layer;
    switch (layerData.GetType()) {
    case TInput:
        layer = shared_ptr<BaseLayer>(new InputLayer(layerData.GetName(), layerData.GetP1(), layerData.GetP2(), layerData.GetP3(), layerData.GetP4()));
        break;
    case TRelu:
        layer = shared_ptr<BaseLayer>(new Relu(layerData.GetName(), m_currentLayer));
        break;
    case TSigmoid:
        layer = shared_ptr<BaseLayer>(new Sigmoid(layerData.GetName(), m_currentLayer));
        break;
    case TFullyConnected:
        layer = shared_ptr<BaseLayer>(new FullyConnected(layerData.GetName(), m_currentLayer,layerData.GetP1()));
        break;
    case TDropout:
        layer = shared_ptr<BaseLayer>(new DropOut(layerData.GetName(), m_currentLayer,layerData.GetPF1()));
        break;
    case TBatchNorm:
        layer = shared_ptr<BaseLayer>(new BatchNorm(layerData.GetName(), m_currentLayer,layerData.GetPF1(), layerData.GetPF2()));
        break;
    case TSoftMax:
        layer = shared_ptr<BaseLayer>(new SoftMax(layerData.GetName(), m_currentLayer));
        break;
    default:
        throw invalid_argument("Layer not implemented.");
        break;
    }
    m_hashMapLayers.insert(make_pair(layerData.GetName(),shared_ptr<BaseLayer>(layer)));
    m_layerNamesList.push_back(layerData.GetName());
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
