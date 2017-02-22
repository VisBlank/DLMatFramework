%% Vanila Max pooling with n-dimensions
% Parameters:
% F: Kernel size
% S: stride

function outConv = avg_pooling_forward(input, Ky,Kx,S)
%% Get the input size in terms of rows and cols
[rowsIn, colsIn, channels, batchSize] = size(input);

%% Initialize outputs to have the same size of the input
sizeRowsOut = ceil((rowsIn-Ky)/S) + 1;
sizeColsOut = ceil((colsIn-Kx)/S) + 1;
outConv = zeros(sizeRowsOut , sizeColsOut, channels, batchSize);

%% Initialize a sampling window
window = [];

%% Sample the input signal to form the window
% Iterate on every element of the batch
for n=1:batchSize
    % Select the current batch item
    inCurrBatchItem = input(:,:,:,n);
    % Iterate on every dimension
    for depth=1:channels
        inCurrDepth = inCurrBatchItem(:,:,depth);
        % Iterate on every element of the input signal
        % Iterate on every row
        for idxRowsIn = 0 : sizeRowsOut - 1
            % Iterate on every col
            for idxColsIn = 0 : sizeColsOut - 1
                % Populate our window (same size of the kernel)
                hstart = (idxRowsIn * S) + 1;
                wstart = (idxColsIn * S) + 1;
                hend = min(hstart + Ky - 1,  rowsIn);
                wend = min(wstart + Kx - 1,  colsIn);
                for idxRowsKernel = 0 : (hend - hstart)
                    for idxColsKernel = 0 : (wend - wstart)
                        window(idxRowsKernel + 1,idxColsKernel + 1) =  inCurrDepth(hstart + idxRowsKernel , wstart + idxColsKernel);
                    end
                end
                % Get the biggest value on the window here...
                outConv(idxRowsIn + 1 , idxColsIn + 1,depth, n) = getMax(window);
                window = [];
            end
        end
    end
end

%% Moving window effect
% The previous inner for loop updates the variables slideRow, and slideCol
% those updates will create the following effect
%
% <<../../docs/imgs/3D_Convolution_Animation.gif>>
%

end

%% Get the biggest value on the (window .* kernel(ones))
% The convolution is all about the sum of product of the window and kernel,
% bye the way this is a dot product
function result = getMax(window)
result = mean2(window);
end