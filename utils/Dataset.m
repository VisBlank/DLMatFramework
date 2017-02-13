classdef (Sealed) Dataset < handle
    %DATASET Utility function to handle datasets.
    % Ex:
    % clear all; clc; load mnist_oficial;
    % data = Dataset(input_train, output_train_labels,1,784,1,1);
    % batch = data.GetBatch(10);
    % display_MNIST_Data(reshape_row_major(batch.X,[10,784]))
    
    properties(Access = private)
        m_X;m_X_val;
        m_Y;m_Y_val;
        m_Y_one_hot;m_Y_val_one_hot;
        m_X_Tensor;m_X_Val_Tensor;
        m_trainingSize;
        m_ValidationSize;
        m_shuffledIndex;m_shuffledIndexVal;
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
        function obj = Dataset(X,Y,rows, cols, channels,dimNumSamples, doOneHot)
            obj.m_X = X;
            obj.m_Y = Y;
            if doOneHot
                obj.m_Y_one_hot = obj.oneHot(Y);
            else
                obj.m_Y_one_hot = Y;
            end
            obj.m_trainingSize = size(X,dimNumSamples);
            
            % Create a shuffled index
            obj.m_shuffledIndex = randperm(obj.m_trainingSize);
            
            % The expected tensor for data is
            % [rows, cols, depth, batch]
            obj.m_X_Tensor = reshape_row_major(X,[rows,cols,channels,obj.m_trainingSize]);
            
            % Transpose rows,cols if image
            % if (rows > 1 && cols > 1)
            %   obj.m_X_Tensor = permute(obj.m_X_Tensor,[2 1 3 4]);
            % end
        end
        
        function AddValidation(obj,X,Y,rows, cols, channels,dimNumSamples, doOneHot)
            obj.m_X_val = X;
            obj.m_Y_val = Y;
            if doOneHot
                obj.m_Y_val_one_hot = obj.oneHot(Y);
            else
                obj.m_Y_val_one_hot = Y;
            end            
            obj.m_ValidationSize = size(X,dimNumSamples);
            
            % Create a shuffled index
            obj.m_shuffledIndexVal = randperm(obj.m_ValidationSize);
            
            % The expected tensor for data is
            % [rows, cols, depth, batch]
            obj.m_X_Val_Tensor = reshape_row_major(X,[rows,cols,channels,obj.m_ValidationSize]);
            
            % Transpose rows,cols if image
            %if (rows > 1 && cols > 1)
            %    obj.m_X_Val_Tensor = permute(obj.m_X_Val_Tensor,[2 1 3 4]);
            %end
        end
        
        function batch = GetBatch(obj,batchSize)            
            selIndex = randperm(obj.m_trainingSize);
            %% TODO: I don't know if I need to reshufle every new batch
            %selIndex = [1:1:batchSize];
            selIndex = selIndex(1:batchSize);
            batch.X = obj.m_X_Tensor(:,:,:,selIndex);
            batch.Y = obj.m_Y_one_hot(selIndex,:);
        end
        
        function batch = GetValidationBatch(obj,batchSize)            
            selIndex = obj.m_shuffledIndexVal(1:batchSize);
            batch.X = obj.m_X_Val_Tensor(:,:,:,selIndex);
            batch.Y = obj.m_Y_val_one_hot(selIndex,:);
        end
        
        function X = GetOriginalInput(obj)
            X = obj.m_X;
        end
        
        function Y = GetOriginalLabels(obj)
            Y = obj.m_Y;
        end
        
        function Y = GetNumClasses(obj)
            Y = numel(unique(obj.m_Y));
        end
        
        function Y = GetOneHotLabels(obj)
            Y = obj.m_Y_one_hot;
        end
        
        function dataSize = GetTrainSize(obj)
            dataSize = obj.m_trainingSize;
        end
        
        function pushToGPU(obj)
            obj.m_X_Tensor = gpuArray(obj.m_X_Tensor);
            obj.m_Y_one_hot = gpuArray(obj.m_Y_one_hot);
            if ~isempty(obj.m_X_Val_Tensor)
                obj.m_X_Val_Tensor = gpuArray(obj.m_X_Val_Tensor);
                obj.m_Y_val_one_hot = gpuArray(obj.m_Y_val_one_hot);
            end
        end
    end
    
end

