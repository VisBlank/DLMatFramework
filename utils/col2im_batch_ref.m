function [image_out] = col2im_batch_ref(cols,H,W,C,N, k_h, k_w, pad, stride)    
    HH = (H+2*pad-k_h)/stride +1;
    WW = (W+2*pad-k_w)/stride +1;
    x_padded = zeros(H+2*pad,W+2*pad,C,N);
    
    for c=1:C
        for ii=1:k_h
            for jj=1:k_w
                row = c * k_w * k_h + ii * k_h + jj;
                for yy=1:HH
                    for xx=1:WW
                        for i=1:N
                            col = yy * WW * N + xx * N + i;
                            x_padded(stride * yy + ii, stride * xx + jj, i, c) = x_padded(stride * yy + ii, stride * xx + jj, i, c) + cols(row, col);
                        end
                    end
                end
            end
        end
    end
    
    if pad > 0
        image_out = x_padded(padding:-padding, padding:-padding, :, :);
    else
        image_out = x_padded;
    end
end