function [image_out] = col2im_ref(mul,height,width,C)
% Convert the column matrix to an image.
F = size(mul,1); %number of filters

if C == 1
    % Single channel out
    image_out = single(zeros(height,width,F));
    for ii = 1:F
        col = mul(ii,:); %change to use ii to go through filters
        res = reshape_row_major(col,[height width]);
        image_out(:,:,ii) = res(:,:,1);
    end
else
    % Multiple channel out
    image_out = single(zeros(height,width,C,F));
    for ii = 1:F
        col = mul(ii,:);
        res = reshape_row_major(col,[height width C]);
        image_out(:,:,ii) = res(:,:,1);
    end
end