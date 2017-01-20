classdef InputLayer < BaseLayer
    %INPUTLAYER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties (Access = 'protected')
        weights
        biases
        activations
        config
        previousInput
        name
        index
        inputLayer = [];
        
        numRows
        numCols
        numChannels
        batchSize
        activationShape
    end
    
    methods (Access = 'public')
        function [obj] = InputLayer(name, numRows,numCols,numChannels,batchSize, index)
            obj.name = name;
            obj.numRows = numRows;
            obj.numCols = numCols;
            obj.numChannels = numChannels;
            obj.batchSize = batchSize;            
            obj.index = index; 
            obj.activationShape = [numRows numCols numChannels batchSize];
        end
                        
        function [activations] = ForwardPropagation(obj, scores, weights, bias)
            activations = obj.activations;
        end
        
        % We don't do backpropagation on the input layer
        function [gradient] = BackwardPropagation(obj, dout)
            gradient = [];
        end
        
        function setActivations(obj, pData)
            % Input X is a row vector
            obj.activations = pData;
        end
        
        function [shapeInput] = getInputShape(obj)
            shapeInput = [obj.numRows,obj.numCols,obj.numChannels,obj.batchSize];
        end
        
        function [actShape] = getActivationShape(obj)
           actShape = obj.activationShape;
       end
    end
    
end

