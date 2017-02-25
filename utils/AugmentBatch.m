classdef AugmentBatch < handle
    %AUGMENTBATCH Utility class to do batch augmentation
    %   Detailed explanation goes here
    
    properties
    end
    
    methods(Access = public)
        function batchOut = Augment(obj,batchIn, cropDims)
            % The augmentation will not change the number of elements on
            % the batch, it will only apply few transformations on it
            % Flip coin
            prob = rand();
            
            % Do random crop disregarding the probability
            if ~isempty(cropDims)
                batchIn = obj.randomCropSize(batchIn, cropDims, true);            
            end
            
            % Choose between one of the augmentations
            if prob >= 0.75
                % Do grayscale conversion
                batchOut = obj.convertToGrayscale(batchIn);
            elseif prob >= 0.5
                % Jitter the colors
                batchOut = obj.colorJittering(batchIn);
            elseif prob >= 0.25
                % Do Sepia filter (old-style image)
                batchOut = obj.sepiaFilter(batchIn);
            else
                % Don't change the batch
                batchOut = batchIn;
            end
            
            % Do horizontal-flip with 0.5 probability (Not related to
            % previous probability)
            batchOut = obj.flip_H_Image(batchOut, 0.5);
        end
    end
    
    methods (Access = public)
        function [batchOut] = convertToGrayscale(obj,batchIn)
            % Select channels
            R = batchIn(:,:,1,:);
            G = batchIn(:,:,2,:);
            B = batchIn(:,:,3,:);
            % Convert to grayscale but keep the number of channels.
            batchOut(:,:,1,:) = 0.2989 * R + 0.5870 * G + 0.1140 * B;
            batchOut(:,:,2,:) = batchIn(:,:,1,:);
            batchOut(:,:,3,:) = batchIn(:,:,1,:);
        end
        
        function [batchOut] = colorJittering(obj, batchIn)
            batchSize = size(batchIn,4);
            for idx = 1:batchSize
                img = batchIn(:,:,:,idx);
                % Channel to jitter (one at a time, or all of them)
                channel = randi([1 4]);
                
                % Use a random value bigger than 0.4 but less then 1
                randVal = 0.4 + rand();
                if randVal > 1
                    randVal = 1;
                end
                
                randVal_R = 0.4 + rand(); randVal_R(randVal_R>1)=1;
                randVal_G = 0.4 + rand(); randVal_G(randVal_G>1)=1;
                randVal_B = 0.4 + rand(); randVal_B(randVal_B>1)=1;
                switch channel
                    case 1
                        batchOut(:,:,:,idx) = imadjust(uint8(img),[0 0 0; randVal 1 1],[]);
                    case 2
                        batchOut(:,:,:,idx) = imadjust(uint8(img),[0 0 0; 1 randVal 1],[]);
                    case 3
                        batchOut(:,:,:,idx) = imadjust(uint8(img),[0 0 0; 1 1 randVal],[]);
                    case 4
                        batchOut(:,:,:,idx) = imadjust(uint8(img),[0 0 0; randVal_R randVal_G randVal_B],[]);
                end
            end
            if isa(batchIn,'single')
                batchOut = single(batchOut);
            else
                batchOut = double(batchOut);
            end
        end
        
        function [batchOut] = sepiaFilter(obj, batchIn)
            % Select channels
            R = batchIn(:,:,1,:);
            G = batchIn(:,:,2,:);
            B = batchIn(:,:,3,:);
            % Convert to grayscale but keep the number of channels.
            batchOut(:,:,1,:) = 0.393 * R + 0.769 * G + 0.189 * B;
            batchOut(:,:,2,:) = 0.349 * R + 0.686 * G + 0.168 * B;
            batchOut(:,:,3,:) = 0.272 * R + 0.534 * G + 0.131 * B;
        end
        
        function [batchOut] = flip_H_Image(obj,batchIn, prob)
            if rand() < prob
                batchOut = flip(batchIn,2);
            else
                batchOut = batchIn;
            end
        end
        
        % Do some rotation on each image of the batch
        function [rotImages] = addRotation(obj, imageIn)
            rotImages = imrotate(imageIn,randi([-8 8]),'crop');
        end
        
        % Add some random noise on the batch
        function [noiseImg] = addPeperNoise(obj, imageIn)
            noiseImg = imnoise(imageIn,'gaussian', 0, 0.01 * rand());
        end
        
        % Another way to create ilumination changes (Simpler than PCA)
        function [chImg] = changeLumination(obj, imageIn)
            % The cie lab color space has better control for illumination
            % than HSL/HSV (But slower to compute)
            lab_image = rgb2lab(imageIn);
            
            channel = 1;
            
            meanValue = mean2(lab_image(:,:,channel));
            
            % Change value for illumination
            lab_image(:,:,channel) = lab_image(:,:,channel) + randi([int32(-meanValue),int32(meanValue)]);
            
            chImg = lab2rgb(lab_image);
        end
        
        function [cropBatch] = randomCropSize(obj, batchIn, cropSize, isTraining)
            [nrows,ncols, channels, batchSize] = size(batchIn);
            displacement = [nrows, ncols] - cropSize;
            
            % Pre-allocate to save dynamic allocation
            cropBatch = zeros([cropSize(1) cropSize(2) channels batchSize], 'like',batchIn);
            
            if isTraining
                % Do random crop (Training)
                for idxBatch=1:batchSize
                    randX = randi(displacement(1));
                    randY = randi(displacement(2));
                    cropBatch(:,:,:,idxBatch) = batchIn(randX:randX+cropSize(1)-1, randY:randY+cropSize(2)-1,:,idxBatch);
                end
            else
                % Crop from center (Test)
                displacement = displacement / 2;
                cropBatch = batchIn(displacement(1):displacement(1)+cropSize(1)-1, displacement(2):displacement(2)+cropSize(2)-1,:,:);
            end
        end                        
    end    
end

