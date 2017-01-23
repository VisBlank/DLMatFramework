classdef (Sealed) Dataset < handle
    %DATASET Utility function to handle datasets.
    % Ex:
    % clear all; clc; load mnist_oficial;
    % data = Dataset(input_train, output_train_labels,1,784,1,1);
    % batch = data.GetBatch(10);
    % display_MNIST_Data(reshape_row_major(batch.X,[10,784]))
    
    properties(Access = private)
        m_X;
        m_Y;
        m_Y_one_hot;
        m_X_Tensor;
        m_trainingSize;
        m_shuffledIndex;
    end
    
    methods(Access = private)
        function oneHotLabels = oneHot(obj,labels)
            valueLabels = unique(labels);
            nLabels = length(valueLabels);
            nSamples = size(labels,1);
            
            oneHotLabels = zeros(nSamples, nLabels);
            
            for i = 1:nLabels
                oneHotLabels(:,i) = (labels == valueLabels(i));
            end
        end
    end
    
    methods(Access = public)
        function obj = Dataset(X,Y,rows, cols, channels,dimNumSamples)
            obj.m_X = X;
            obj.m_Y = Y;
            obj.m_Y_one_hot = obj.oneHot(Y);
            obj.m_trainingSize = size(X,dimNumSamples);
            
            % Create a shuffled index
            obj.m_shuffledIndex = randperm(obj.m_trainingSize);
            
            % The expected tensor for data is
            % [rows, cols, depth, batch]
            obj.m_X_Tensor = reshape_row_major(X,[rows,cols,channels,obj.m_trainingSize]);
        end
        
        function batch = GetBatch(obj,batchSize)
            selIndex = obj.m_shuffledIndex(1:batchSize);
            batch.X = obj.m_X_Tensor(:,:,:,selIndex);
            batch.Y = obj.m_Y_one_hot(selIndex,:);
        end
        
        function X = GetOriginalInput(obj)
            X = obj.m_X;
        end
        
        function Y = GetOriginalLabels(obj)
            Y = obj.m_Y;
        end
        
        function Y = GetOneHotLabels(obj)
            Y = obj.m_Y_one_hot;
        end
    end
    
end

