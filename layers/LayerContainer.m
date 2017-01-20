classdef LayerContainer < handle
    % Container used to store all layers used on the model
    % References:
    % https://uk.mathworks.com/matlabcentral/fileexchange/25024-data-structure--a-cell-array-list-container
    
    % Ex:
    % cont = LayerContainer();    
    % cont <= struct('name','ImageIn','type','input','rows',1,'cols',2,'depth',1, 'batchsize',1);
    % cont <= struct('name','FC_1','type','fc');
    % cont.getNumLayers()
    
    properties (Access = private)
        % It will be a cell there is no list on matlab
        layersContainer = containers.Map('KeyType','char','ValueType','any');
        layersCellContainer = {};
        numLayers = 0;
    end
    
    methods (Access = 'protected')
        function pushLayer(obj,metaDataLayer)
            switch metaDataLayer.type                
                case 'input'                    
                    layerInst = InputLayer(metaDataLayer.name, metaDataLayer.rows,metaDataLayer.cols,metaDataLayer.depth, metaDataLayer.batchsize, obj.numLayers+1);
                case 'fc'                    
                    layerInst = FullyConnected(metaDataLayer.name, obj.numLayers+1);                
                case 'relu'
                    layerInst = Relu(metaDataLayer.name, obj.numLayers+1);                                                
                case 'softmax'
                    layerInst = Softmax(metaDataLayer.name, obj.numLayers+1);
                otherwise
                    fprintf('Layer %s not implemented\n',metaDataLayer.type);
            end
            
            % Handle objects are copied as reference
            obj.layersContainer(metaDataLayer.name) = layerInst;
            obj.layersCellContainer{obj.numLayers+1} = layerInst;
            obj.numLayers = obj.numLayers + 1;
        end
    end
    
    methods (Access = 'public')
        function obj = LayerContainer()            
            obj.numLayers = 0;
        end
        
        % Override the "<=" operator (used to push a new layer)
        function result = le(obj,B)
            obj.pushLayer(B);
            result = [];
        end
        
        function removeLayer(obj,name)
            obj.numLayers = obj.numLayers - 1;
            remove(obj.layersContainer,name);
        end
        
        function layer = getLayerFromName(obj,name)            
            layer = obj.layersContainer(name);
        end
        
        function layer = getLayerFromIndex(obj,index)            
            layer = obj.layersCellContainer(index);
        end
        
        function numLayers = getNumLayers(obj)
            numLayers = obj.layersContainer.Count;
        end
                
        
        function ShowStructure(obj)
            % Iterate on all layers            
            for idxLayer=1:obj.layersContainer.Count
                layerInstance = obj.layersCellContainer{idxLayer};
                txtDesc = layerInstance.getName();
                fprintf('LAYER(%d)--> %s\n',layerInstance.getIndex(),txtDesc);
            end
        end
        
        function generateGraphVizStruct(obj)
            
        end
    end
end