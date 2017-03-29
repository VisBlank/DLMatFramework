classdef (Sealed) Dataset < handle
    %DATASET Utility class to handle datasets.
    % Ex:
    % clear all; clc; load mnist_oficial;
    % data = Dataset(input_train, output_train_labels,1,784,1,1);
    % batch = data.GetBatch(10);
    % display_MNIST_Data(reshape_row_major_custom(batch.X,[10,784]))
    
    properties(Access = private)
        m_X;m_X_val;
        m_Y;m_Y_val;
        m_Y_one_hot;m_Y_val_one_hot;
        m_X_Tensor;m_X_Val_Tensor;
        m_trainingSize;
        m_inputAreImages;
        m_ValidationSize;
        m_shuffledIndex;m_shuffledIndexVal;
        m_doAugmentation = false;
        m_cropDims = [];
        m_doMeanPixelNorm = false;
        m_doMeanImageNorm = false;
        m_meanPixVals;
        m_meanImage;
        m_augmenter;
        hasValidation = false;
        batchPosition = 1;
        iterationCounter = 0;
        shuffleTime = 0;
        indexShuffle;
    end
            
    methods(Static)
        function obj = loadFromHDF5(filename)
            obj = [];
            % load the hdf5 file into the data class
            obj.m_X = h5read(filename,'/m_X');
            obj.m_Y = h5read(filename,'/m_Y');
            obj.m_Y_one_hot = h5read(filename,'/m_Y_onehot');
            obj.m_X_Tensor = h5read(filename,'/m_X_Tensor');
            try
                obj.m_X_val = h5read(filename,'/m_X_val');
            end
            try
                obj.m_Y_val = h5read(filename,'/m_Y_val');
            end
            try
                obj.m_Y_val_one_hot = h5read(filename,'/m_Y_val_one_hot');
            end
            try
                obj.m_X_Val_Tensor = h5read(filename, 'm_X_Val_Tensor');
            end
            
            %obj = Dataset(m_X,m_Y,size(),size(m_X,2),size(m_Y,1),size(m_X,3),true);
        end
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
        
        function batchOut = NormalizeImage(obj, batchIn)
            if obj.m_doMeanPixelNorm
                batchOut(:,:,1,:) = batchIn(:,:,1,:) - obj.m_meanPixVals{1};
                batchOut(:,:,2,:) = batchIn(:,:,2,:) - obj.m_meanPixVals{2};
                batchOut(:,:,3,:) = batchIn(:,:,3,:) - obj.m_meanPixVals{3};
            end
        end
    end
    
    methods(Access = public)
        function obj = Dataset(X,Y,rows, cols, channels,dimNumSamples, doOneHot)
            obj.m_X = X;
            obj.m_Y = Y;
            obj.hasValidation = false;
            obj.m_augmenter = AugmentBatch();
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
            obj.m_X_Tensor = reshape_row_major_custom(X,[rows,cols,channels,obj.m_trainingSize]);
            
            % Detect if we're dealing with images 
            if (rows > 1 && cols > 1)
                obj.m_inputAreImages = true;
                % Transpose rows,cols if image
                %obj.m_X_Tensor = permute(obj.m_X_Tensor,[2 1 3 4]);
            end
        end
        
        function AddValidation(obj,X,Y,rows, cols, channels,dimNumSamples, doOneHot)
            obj.m_X_val = X;
            obj.m_Y_val = Y;
            obj.hasValidation = true;
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
            obj.m_X_Val_Tensor = reshape_row_major_custom(X,[rows,cols,channels,obj.m_ValidationSize]);
            
            % Transpose rows,cols if image
            %if (rows > 1 && cols > 1)
            %    obj.m_X_Val_Tensor = permute(obj.m_X_Val_Tensor,[2 1 3 4]);
            %end
        end
        
        function enableAugmentation(obj, flag)
            obj.m_doAugmentation = flag;
        end
        
        function doCrop(obj, cropDims)
           obj.m_cropDims = cropDims;
        end
        
        function enableMeanPixelNormalization(obj, flag, pixVals)
            obj.m_doMeanPixelNorm = flag;
            obj.m_meanPixVals = pixVals;
        end
        
        function enableMeanImageNormalization(obj, flag, pixVals)
            obj.m_doMeanImageNorm = flag;
            obj.m_meanImage = pixVals;
        end
        
        function batch = GetBatch(obj,batchSize)
            % Select the whole dataset if batchSize is negative
            if (batchSize <= 0)
                batchSize = obj.m_trainingSize;
            end
            
            % Shuffle dataset on first iteration or if reaching set iter.
            % Then reset the counter for place in the batch and set iter
            if (obj.iterationCounter == 0 || obj.iterationCounter == obj.shuffleTime)
                obj.indexShuffle = randperm(obj.m_trainingSize);
                obj.batchPosition = 1;
                obj.iterationCounter = 0;
            end
            selIndex = obj.indexShuffle;
            
            % Check if batch will take more elements than are left in the
            % dataset to take
            if (obj.batchPosition + batchSize -1  > size(selIndex,2))
                remainingBatchSize = batchSize - length(selIndex(obj.batchPosition:end));
                selIndex = [selIndex(obj.batchPosition:end) selIndex(1:remainingBatchSize)];
                obj.batchPosition = 1;
                warning('Not enough data left for complete batch')
            else
                selIndex = selIndex(obj.batchPosition:obj.batchPosition + batchSize - 1);
                obj.batchPosition = obj.batchPosition + batchSize;
            end
            
            batch.X = obj.m_X_Tensor(:,:,:,selIndex);
            batch.Y = obj.m_Y_one_hot(selIndex,:);
            
            % Do augmentation if needed
            if obj.m_doAugmentation
                batch.X = obj.m_augmenter.Augment(batch.X, obj.m_cropDims);
            end
            
            % Do Normalization if needed
            if obj.m_doMeanImageNorm || obj.m_doMeanPixelNorm
                batch.X = obj.NormalizeImage(batch.X);
            end
            
            obj.iterationCounter = obj.iterationCounter + 1;
        end
        
        function batch = GetValidationBatch(obj,batchSize)
            % Select the whole dataset if batchSize is negative
            if (batchSize <= 0)
                batchSize = obj.m_ValidationSize;
            end
            selIndex = obj.m_shuffledIndexVal(1:batchSize);
            batch.X = obj.m_X_Val_Tensor(:,:,:,selIndex);
            batch.Y = obj.m_Y_val_one_hot(selIndex,:);
            
            % Do Normalization if needed
            if obj.m_doMeanImageNorm || obj.m_doMeanPixelNorm
                batch.X = obj.NormalizeImage(batch.X);
            end
            
            % Get Crop (Central crop if needed)
            if ~isempty(obj.m_cropDims)
                batch.X = obj.m_augmenter.randomCropSize(batch.X, obj.m_cropDims, false);
            end
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
        
        function flag = HasValidation(obj)
           flag = obj.hasValidation; 
        end
        
        function shuffleEveryNIterations(obj, numIterations)
            obj.shuffleTime = numIterations;
        end
        
        function pushToGPU(obj)
            obj.m_X_Tensor = gpuArray(obj.m_X_Tensor);
            obj.m_Y_one_hot = gpuArray(obj.m_Y_one_hot);
            if ~isempty(obj.m_X_Val_Tensor)
                obj.m_X_Val_Tensor = gpuArray(obj.m_X_Val_Tensor);
                obj.m_Y_val_one_hot = gpuArray(obj.m_Y_val_one_hot);
            end
        end
        
        function saveToHDF5(obj,filename)
            % set a default filename if not supplied
            if ~exist('filename','var')
                filename = 'dataset.h5';
            end
            % get the data and create hdf5 file
            m_X_hdf = obj.m_X;
            m_Y_hdf = obj.m_Y;
            m_Y_onehot_hdf = obj.m_Y_one_hot;
            m_X_Tensor_hdf = obj.m_X_Tensor;
            if ~isempty(obj.m_X_val)
                m_X_val_hdf = obj.m_X_val;
                h5create(filename,'/m_X_val',size(m_X_val_hdf));
            end
            if ~isempty(obj.m_Y_val)
                m_Y_val_hdf = obj.m_Y_val;
                h5create(filename,'/m_Y_val',size(m_Y_val_hdf));
            end
            if ~isempty(obj.m_Y_val_one_hot)
                m_Y_val_one_hot_val_hdf = obj.m_Y_val_one_hot;
                h5create(filename,'/m_Y_val_one_hot',size(m_Y_val_one_hot_val_hdf));
            end
            if ~isempty(obj.m_X_Val_Tensor)
                m_X_Val_Tensor_hdf = obj.m_X_Val_Tensor;
                h5create(filename,'/m_X_Val_Tensor',size(m_X_Val_Tensor_hdf));
            end
            h5create(filename,'/m_X',size(m_X_hdf));
            h5create(filename,'/m_Y',size(m_Y_hdf));
            h5create(filename,'/m_Y_onehot',size(m_Y_onehot_hdf));
            h5create(filename,'/m_X_Tensor',size(m_X_Tensor_hdf));
            
            % write data to the hdf5 file
            h5write(filename,'/m_X',m_X_hdf);
            h5write(filename,'/m_Y',m_Y_hdf);
            h5write(filename,'/m_Y_onehot',m_Y_onehot_hdf);
            h5write(filename,'/m_X_Tensor',m_X_Tensor_hdf);
            if ~isempty(obj.m_X_val)
                h5write(filename,'/m_X_val',size(m_X_val_hdf));
            end
            if ~isempty(obj.m_Y_val)
                h5write(filename,'/m_Y_val',size(m_Y_val_hdf));
            end
            if ~isempty(obj.m_Y_val_one_hot)
                h5write(filename,'/m_Y_val_one_hot',size(m_Y_val_one_hot_val_hdf));
            end
            if ~isempty(obj.m_X_Val_Tensor)
                h5write(filename,'/m_X_Val_Tensor',size(m_X_Val_Tensor_hdf));
            end
        end
    end
    
end

