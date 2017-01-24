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
            % Unless pre-defined the previous layer will be the input of
            % the current layer.
            if isfield(metaDataLayer, 'inputLayer')
                previousLayer = obj.layersContainer(name);
            else
                if obj.numLayers == 0
                    previousLayer = [];
                else
                    previousLayer = obj.layersCellContainer{obj.numLayers};
                end
            end
            switch metaDataLayer.type                
                case 'input'                    
                    layerInst = InputLayer(metaDataLayer.name, metaDataLayer.rows,metaDataLayer.cols,metaDataLayer.depth, metaDataLayer.batchsize, obj.numLayers+1);
                case 'fc'                    
                    layerInst = FullyConnected(metaDataLayer.name, metaDataLayer.num_output, obj.numLayers+1, previousLayer);                
                case 'relu'
                    layerInst = Relu(metaDataLayer.name, obj.numLayers+1, previousLayer);                                                
                case 'sigmoid'
                    layerInst = Sigmoid(metaDataLayer.name, obj.numLayers+1, previousLayer);
                case 'softmax'
                    layerInst = Softmax(metaDataLayer.name, obj.numLayers+1, previousLayer);
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
            layer = layer{1};
        end
        
        function numLayers = getNumLayers(obj)
            numLayers = obj.layersContainer.Count;
        end
        
        function cellLayers = getAllLayers(obj)
            cellLayers = obj.layersContainer.values;
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