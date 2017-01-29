function [image_out] = col2im_batch_ref(mul,height,width,C,N)

F = size(mul,1); %number of filters
nPatch = size(mul,2)/N;

if C == 1
    image_out = zeros(height,width,F);
    
    for n=1:N
        for ii = 1:F
            col = mul(ii,n*nPatch-nPatch+1:n*nPatch);
            image_out(:,:,ii,n) = reshape_row_major(col,[height width]);
        end
    end
    
else
    image_out = zeros(height,width,C,F);
    for n=1:N
        for ii = 1:F
            col = mul(ii,n*nPatch-nPatch+1:n*nPatch);
            image_out(:,:,ii,n) = reshape_row_major(col,[height width C 1]);
        end
    end
    
end