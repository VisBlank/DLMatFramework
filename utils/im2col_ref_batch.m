function [ img_matrix ] = im2col_ref_batch( inputImg, k_height, k_width, S , P, isConv)
%IM2COL Convert image to a matrix, this step is used to accelerate
%convolutions, implementing the convolution as a matrix multiplication
% Rgb image is a 3d matrix [rows,cols,color]
% This version currently does not support batch of images, we choose this
% because we're first going to use the CPU mode, and we want to relly on
% parfor (OpenMP)
coder.extrinsic('warning');

% Get Image dimensions
[imgHeight, imgWidth, imgChannels, imgN] = size(inputImg);

% Calculate convolved result size.
newImgHeight = ((imgHeight + 2*P - k_height) / S)+1;
newImgWidth = ((imgWidth + 2*P - k_width) / S)+1;

incorrectConvParam = 0; incorrectPoolParam = 0;
offsetHeight = 0;
offsetWidth = 0;

% Check if it is a real number
if rem(newImgHeight,1) ~= 0 || rem(newImgWidth,1) ~= 0
    warning('warning: Invalid stride or pad for input\n');
    if isConv
        % Convolution do a floor
        newImgHeightFl = floor(((imgHeight + 2*P - k_height) / S)+1);
        newImgWidthFl = floor(((imgWidth + 2*P - k_width) / S)+1);
        incorrectConvParam = 1;
    else
        % Pooling do a ceil and adapt the sampling window
        newImgHeightCe = ceil(((imgHeight + 2*P - k_height) / S)+1);
        newImgWidthCe = ceil(((imgWidth + 2*P - k_width) / S)+1);
        % How much each side should grow to
        growHeight = imgHeight + (mod(imgHeight - k_height, S));
        growWidth = imgWidth + (mod(imgWidth - k_width, S));
        
        
        growHeightdif = growHeight - imgHeight;
        growWidthdif = growWidth - imgWidth;
        % If the error is bigger than 0.5 Don't accept
        error_Height = newImgHeightCe - newImgHeight;
        error_Width = newImgWidthCe - newImgWidth;
        if error_Height > 0.5 || error_Width > 0.5
           error('Invalid pooling calculation, error bigger than 0.5\n');
        else
            warning('Input size is %dx%d and should be %dx%d\n',imgHeight,imgWidth,imgHeight+growHeightdif, imgWidth+growWidthdif);
            warning('Kernel sizes: %dx%d Stride: %d\n',k_height,k_width,S);
            newImgHeight = newImgHeightCe;
            newImgWidth = newImgWidthCe;
        end
        
        incorrectPoolParam = 1;
        
        % Calculate how much we need to offset out image end point for window scanning 
        offsetHeight = (growHeight - S + 1) - imgHeight;
        offsetWidth = (growWidth - S + 1) - imgWidth;
        warning('Height offset: %d Width offset:%d\n',offsetHeight, offsetWidth);
    end
end

% Allocate output sizes
img_matrix = single(zeros(...
    (imgChannels*k_height*k_width),(newImgHeight * newImgWidth * imgN) ...
    ));

% Only pad if needed
if P ~= 0
    inputImg = padarray(inputImg,[P P]);
    % Get dimensions again before iterate on padded image, otherwise we will
    % keep sampling with the old (unpadded size)
    [imgHeight, imgWidth, ~] = size(inputImg);
end

% Iterate on the input image like a convolution
cont = 1;
for n=1:imgN
    for r=1:S:(imgHeight+offsetHeight) %last possible starting point
        for c=1:S:(imgWidth+offsetWidth) %last possible starting point
            % Avoid slide out of the image (Security buffer overflow)
            if (((c+k_width)-1) <= imgWidth) && (((r+k_height)-1) <= imgHeight)
                % Select window on input volume
                patch = inputImg(r:(r+k_height)-1,c:(c+k_width)-1,:,n);
                
                % Convert patch to a col vector, the matlab reshape order is
                % row major while other languages (C/C++, python) are col
                % major, on this particular case (im2col, then matrix mult with
                % the kernel) this order will not mather, but it's not allways
                % true...
                patchRow = reshape(patch,[],1);
                patch = [];
                % Append the transformed patch into the output matrix
                img_matrix(:,cont) = patchRow;
                cont = cont+1;
            else
                if incorrectPoolParam
                    % Sliding window is partially outside the window image,
                    % on this case we sample with all that is still available
                    if ((r+k_height)-1) >= imgHeight
                        outVertically = 1;
                    else
                        outVertically = 0;
                    end
                    if ((c+k_width)-1) >= imgWidth
                        outHorizontally = 1;
                    else
                        outHorizontally = 0;
                    end
                    patch_inflated = single(zeros(k_height,k_width,imgChannels));
                    if outHorizontally && ~outVertically
                        % Sample will be a col vector
                        patch = inputImg(r:(r+k_height)-1,c:end,:,n);
                        patch_inflated(:,1:size(patch,2),:) = patch;
                    end
                    if ~outHorizontally && outVertically
                        % Sample will be a row vector
                        patch = inputImg(r:end,c:(c+k_width)-1,:,n);
                        patch_inflated(1:size(patch,1),:,:) = patch;
                    end
                    if outHorizontally && outVertically
                        % Sample will be a scalar
                        patch = inputImg(r:end,c:end,:,n);
                        patch_inflated(1:size(patch,1),1:size(patch,2),:) = patch;
                    end                    
                    % Convert patch to a col vector, the matlab reshape order is
                    % row major while other languages (C/C++, python) are col
                    % major, on this particular case (im2col, then matrix mult with
                    % the kernel) this order will not mather, but it's not allways
                    % true...
                    patchRow = reshape(patch_inflated,[],1);
                    
                    % Append the transformed patch into the output matrix
                    img_matrix(:,cont) = patchRow;
                    cont = cont+1;
                end
            end
        end
    end
end
end