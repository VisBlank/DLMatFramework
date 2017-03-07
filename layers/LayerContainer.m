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
                case 'conv'                    
                    layerInst = ConvolutionLayer(metaDataLayer.name, metaDataLayer.kh, metaDataLayer.kw, metaDataLayer.stride, metaDataLayer.pad, metaDataLayer.num_output, obj.numLayers+1, previousLayer);                
                case 'maxpool'                    
                    layerInst = MaxPoolLayer(metaDataLayer.name, metaDataLayer.kh, metaDataLayer.kw, metaDataLayer.stride, obj.numLayers+1, previousLayer);
                case 'avgpool'                    
                    layerInst = AvgPoolLayer(metaDataLayer.name, metaDataLayer.kh, metaDataLayer.kw, metaDataLayer.stride, obj.numLayers+1, previousLayer);
                case 'add'     
                    previousLayers = obj.getLayersFromName(metaDataLayer.inputs);
                    layerInst = EltWiseAdd(metaDataLayer.name, obj.numLayers+1, previousLayers);
                case 'relu'
                    layerInst = Relu(metaDataLayer.name, obj.numLayers+1, previousLayer);                                                
                case 'dropout'
                    layerInst = Dropout(metaDataLayer.name, metaDataLayer.prob, obj.numLayers+1, previousLayer);                                                
                case 'batchnorm'
                    layerInst = BatchNorm(metaDataLayer.name, metaDataLayer.eps, metaDataLayer.momentum, obj.numLayers+1, previousLayer);                                                
                case 'sp_batchnorm'
                    layerInst = SpatialBatchNorm(metaDataLayer.name, metaDataLayer.eps, metaDataLayer.momentum, obj.numLayers+1, previousLayer);
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
        
        function layers = getLayersFromName(obj, name)
           % On this case name comes from a cell array
           layers = cell(1,numel(name));
           for idxLayer=1:numel(layers)
                layers{idxLayer} = obj.getLayerFromName(name{idxLayer});
           end
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
        
        function layerCont = getLayerMap(obj)
            layerCont = obj.layersContainer;
        end                
        
        function ShowStructure(obj)
            % Iterate on all layers            
            for idxLayer=1:obj.layersContainer.Count
                layerInstance = obj.layersCellContainer{idxLayer};
                txtDesc = layerInstance.getName();
                fprintf('LAYER(%d)--> %s num_inputs:%d [',layerInstance.getIndex(),txtDesc, layerInstance.GetNumInputs());                
                inLayer = layerInstance.getInputLayer();
                if layerInstance.GetNumInputs() < 2                    
                    if ~isempty(inLayer)
                        fprintf(' %s,',inLayer.getName());
                    end
                else
                    for idxIn=1:layerInstance.GetNumInputs()                       
                        fprintf(' %s,',inLayer{idxIn}.getName());
                    end
                end                
                fprintf(']\n');
            end
        end
        
        function dotGraph = generateDotGraph(obj)
            
            dotGraph = {sprintf('digraph CNN {\n')};
            
            for idxLayer=1:obj.layersContainer.Count
                layerInstance = obj.layersCellContainer{idxLayer};
                dotGraph{end+1} = sprintf('%s\n',layerInstance.getName());
            end
            
            for idxLayer=1:obj.layersContainer.Count
                
                layerInstance = obj.layersCellContainer{idxLayer};
                inLayers = layerInstance.getInputLayer();
                if layerInstance.GetNumInputs() < 2                    
                    if ~isempty(inLayers)
                        thisEdge = sprintf('%s->%s\n', inLayers.getName(), layerInstance.getName());
                        dotGraph{end+1} = thisEdge;
                    end
                else
                    for idxIn=1:layerInstance.GetNumInputs()
                        thisEdge = sprintf('%s->%s\n',inLayers{idxIn}.getName(), layerInstance.getName());
                        dotGraph{end+1} = thisEdge;
                    end
                end
            end
            dotGraph{end+1} = sprintf('}');
            dotGraph  = cat(2,dotGraph{:});
            dotFileName = 'graph.dot';
            
            f = fopen(dotFileName, 'w') ; fwrite(f, dotGraph) ; fclose(f) ;
            
            pdfFileName = 'graph.pdf';
            if (exist('dot'))
                cmd = sprintf('dot -Tpdf %s -o %s', dotFileName, pdfFileName) ;
                system(cmd);

                
                switch computer
                    case {'PCWIN64', 'PCWIN'}
                        system(sprintf('start "" "%s"', pdfFileName)) ;
                    case 'MACI64'
                        system(sprintf('open "%s"', pdfFileName)) ;
                    case 'GLNXA64'
                        system(sprintf('display "%s"', pdfFileName)) ;
                    otherwise
                        fprintf('The figure saved at "%s"\n', pdfFileName) ;
                end
            end
        end
    end
end