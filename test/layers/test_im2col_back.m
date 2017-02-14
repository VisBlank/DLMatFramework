clc; clear all;
load im2col_back_data;

dx_padded_res = im2col_back_ref(grad_before_im2col,5, 5, 1, 3, 3, 3); 

% Compare differences
diff = sum(abs(dx_padded_res(:) - dx_padded(:)));
if diff > 1e-9
    error('im2col_back failed');
else
    fprintf('im2col_back passed\n');    
end

dx_padded_res_simple = im2col_back_ref(grad_before_im2col_simple,5, 5, 1, 3, 3, 3); 