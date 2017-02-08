function [ img_matrix ] = im2col_ref( inputImg, k_height, k_width, S , P, isConv )
%IM2COL Convert image to a matrix, this step is used to accelerate
%convolutions, implementing the convolution as a matrix multiplication

% This version currently does not support batch of images, we choose this
% because we're first going to use the CPU mode, and we want to relly on
% parfor (OpenMP)
coder.extrinsic('warning')
% Get Image dimensions
[imgHeight, imgWidth, imgChannels] = size(inputImg);

% Calculate convolved result size.
newImgHeight = ((imgHeight + 2*P - k_height) / S)+1;
newImgWidth = ((imgWidth + 2*P - k_width) / S)+1;

% Check if it is a real number
if rem(newImgHeight,1) ~= 0 || rem(newImgWidth,1) ~= 0
    warning('warning: Invalid stride or pad for input\n');
    if isConv
        % Convolution do a floor
        newImgHeight = floor(((imgHeight + 2*P - k_height) / S)+1);
        newImgWidth = floor(((imgWidth + 2*P - k_width) / S)+1);
    else
        % Pooling do a ceil and adapt the sampling window
        newImgHeight = ceil(((imgHeight + 2*P - k_height) / S)+1);
        newImgWidth = ceil(((imgWidth + 2*P - k_width) / S)+1);
    end
end

% Detect fractional size convolution (Caffe feature)
if  rem(newImgHeight,1) == 0 && rem(newImgWidth,1) == 0
    isFract = 0;
else
    isFract = 1;
end

% Calculate biggest row/col size considering padding or fractional convolution
maxHeight = imgHeight - isFract + ((2*P) );
maxWidth = imgWidth - isFract + ((2*P) );

% Calculate how tall the collumn will be
cols_height = imgChannels *  k_height * k_width;
kernelProd = k_height * k_width;

% Create variables to support richard formula, that calculates idxCol_out
% from (n_rows,n_cols,row,col,ky,kx,stride,width_col)
% Number of collumns that your slide window will cross on
% each dimension.
% Product k(x,y) * stride, calculating outside to avoid this multiplication
% for every row,col,channel       
n_cols = newImgWidth * k_width;
n_rows = newImgHeight * k_height;
prod_kx_stride = k_width * S;
prod_ky_stride = k_height * S;

% Allocate output (use the like because of tests with fixed-point toolbox)
img_matrix = zeros(...
    (imgChannels*k_height*k_width),(newImgHeight * newImgWidth ), ...
    'like',inputImg);

% Iterate on the input image (could be virtually padded)
for channel = 1:imgChannels
    % Move down on the image
    for row = 1:S:imgWidth + (2 * P)
        % Move left on the image
        for col = 1:S:imgHeight + (2 * P)
            % If the window is out of the image we should ignore the current
            % iteration. But take care that this also may happen when we have
            % padding, so check. Because if even with padding we go out of the
             % window we should ignore(continue).
            if ((row + (k_height-1) ) >  maxHeight) || ((col + (k_width - 1)) > maxWidth)
                continue;
            end
            
            % Position the row of output channel related to the current channel
            idxRow_out = (channel - 1) * kernelProd + 1;
            
            % X coordinate on output 2d matrix (Move right on output matrix)
            % Richard formula to calculate the collumn position of the output matrix
            % given (n_rows,n_cols,row,col,ky,kx,stride,width_col). Previously this
            % was calculated as "idxCol_out = (idxCol_out + 1) % with_data_col;"
            % after each window slide, but this was breaking full parallelization.
            idxCol_out = ((n_rows -(n_rows - (row-1)*k_height))/(prod_ky_stride))*newImgWidth + ((n_cols - (n_cols - (col-1)*k_width)))/(prod_kx_stride);
            
            % Select window [ky x kx] on input volume on each channel
            for m = 0:k_height - 1
                for n = 0:k_width - 1
                    % Fix offset if we're doing padding
                    row_pad = (row + n) - P;
                    col_pad = (col + m) - P;
                    
                    % Avoid running out of input image boundaries
                    if ((row_pad > 0) && (col_pad > 0) && (row_pad <= imgHeight) && (col_pad <= imgWidth))
                        img_matrix( idxCol_out * cols_height + (idxRow_out ) ) = inputImg( ((col_pad-1) * imgHeight) + (row_pad ) + ((channel-1) * imgHeight *imgWidth ));
                    else
                        img_matrix( idxCol_out * cols_height + idxRow_out ) = 0;
                    end
                    
                    % Move down on the output 2d array to add current
                    % element from the patch
                    idxRow_out = idxRow_out + 1;
                end
            end
        end
    end
end