clc; clear all;
load im2col_back_data;

%% Check matlab version

dx_padded_res = im2col_back_ref(grad_before_im2col,5, 5, 1, 3, 3, 3); 

% Compare differences
diff = sum(abs(dx_padded_res(:) - dx_padded(:)));
if diff > 1e-9
    error('im2col_back failed');
else
    fprintf('im2col_back passed\n');    
end

dx_padded_res_simple = im2col_back_ref(grad_before_im2col_simple,5, 5, 1, 3, 3, 3); 



%% Check mex c++ version

[dx_padded_res_mex,t0,t1] = mex_im2col_back(grad_before_im2col,5, 5, 1, 3, 3, 3); 

% Compare differences
diff = sum(abs(dx_padded_res_mex(:) - dx_padded(:)));
if diff > 1e-9
    error('im2col_back_mex failed');
else
    fprintf('im2col_back_mex passed\n');    
end

dx_padded_res_simple = im2col_back_ref(grad_before_im2col_simple,5, 5, 1, 3, 3, 3); 