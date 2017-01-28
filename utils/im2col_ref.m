function [ img_matrix ] = im2col_ref( inputImg, k_height, k_width, S , P, isConv )
%IM2COL Convert image to a matrix, this step is used to accelerate
%convolutions, implementing the convolution as a matrix multiplication
% Rgb image is a 3d matrix [rows,cols,color]
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

if  rem(newImgHeight,1) == 0 && rem(newImgWidth,1) == 0
    isFract = 0;
else
    isFract = 1;
end

maxHeight = imgHeight - isFract + ((2*P) );
maxWidth = imgWidth - isFract + ((2*P) );

cols_height = imgChannels *  k_height * k_width;
kernelProd = k_height * k_width;

n_cols = newImgWidth * k_width;
n_rows = newImgHeight * k_height;
prod_kx_stride = k_width * S;
prod_ky_stride = k_height * S;

img_matrix = zeros(...
    (imgChannels*k_height*k_width),(newImgHeight * newImgWidth ), ...
    'like',inputImg);

for channel = 1:imgChannels
    
    for row = 1:S:imgWidth + (2 * P)
        
        for col = 1:S:imgHeight + (2 * P)
            
            if ((row + (k_height-1) ) >  maxHeight) || ((col + (k_width - 1)) > maxWidth)
                continue;
            end
            
            idxRow_out = (channel - 1) * kernelProd + 1;
            
            idxCol_out = ((n_rows -(n_rows - (row-1)*k_height))/(prod_ky_stride))*newImgWidth + ((n_cols - (n_cols - (col-1)*k_width)))/(prod_kx_stride);
            
            for m = 0:k_height - 1
                for n = 0:k_width - 1
                    
                    row_pad = (row + n) - P;
                    col_pad = (col + m) - P;
                    
                    if ((row_pad > 0) && (col_pad > 0) && (row_pad <= imgHeight) && (col_pad <= imgWidth))
                        img_matrix( idxCol_out * cols_height + (idxRow_out ) ) = inputImg( ((col_pad-1) * imgHeight) + (row_pad ) + ((channel-1) * imgHeight *imgWidth ));
                    else
                        img_matrix( idxCol_out * cols_height + idxRow_out ) = 0;
                    end
                    idxRow_out = idxRow_out + 1;
                end
            end
        end
    end
end