function [ img_grad ] = im2col_back_ref( dout, img_grad_before_pad_H, img_grad_before_pad_W, S, kH, kW, C )
%IM2COL_BACK_REF Backpropagation of im2col
% dout: Input of im2col backprop
% Return
% Im2col gradient w.r.t to input times dout

% Calculate the spatial dimensions of im_grad
% Observe that the result will be "padded"
H = (img_grad_before_pad_H - 1) * S + kH;
W = (img_grad_before_pad_W - 1) * S + kW;

% Start with zeros
img_grad = zeros(H,W,C);

% Iterate on all the rows of dout
for ii=1:(img_grad_before_pad_H*img_grad_before_pad_W)
    % Select row from dout
    row = dout(ii,:);
    
    % Create a patch from the row
    patch = reshape_row_major(row,[kH kW C]);
    %patch = reshape(row,[HH WW C]);
    
    % Calculate indexes on dx
    h_start = floor(((ii-1) / img_grad_before_pad_W) * S);    
    w_start = mod((ii-1),img_grad_before_pad_W) * S;
    
    % We increment due to the fact that matlab arrays start at 1
    h_start = h_start + 1;
    w_start = w_start + 1;
    
    % This decrement will not happen on C
    h_end = h_start+kH-1;
    w_end = w_start+kW-1;
        
    % Place/Accumulate patch on img_grad (It's going to sum when 2 patches
    % overlap)
    img_grad(h_start:h_end, w_start:w_end, :) = img_grad(h_start:h_end, w_start:w_end, :) + patch;    
end

end

