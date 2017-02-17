%% Test opencl Relu
clear all; clc; close all;

%% Small matrix test 1 (CPU)
A = single([1 2 3; 4 5 6]);
desired_shape = [6 1 1 1];
[A_reshaped,execTime,trfTime] = mex_reshape_row_major(A,desired_shape);
A_reshaped_ref = reshape_row_major(A, desired_shape);

% Compare results
diff = sum(abs(A_reshaped_ref(:) - A_reshaped(:)));
if (diff < 0.001)
    disp('Reshape row-major test 1 worked (CPU)');
    fprintf('Processing time (trf+exec): %f\n',execTime+trfTime);
else
    error('Reshape row-major test 1 failed (CPU)');
end

%% Small matrix test 2 (CPU)
A = single([1 2 3; 4 5 6]);
desired_shape = [1 6 1 1];
[A_reshaped,execTime,trfTime] = mex_reshape_row_major(A,desired_shape);
A_reshaped_ref = reshape_row_major(A, desired_shape);

% Compare results
diff = sum(abs(A_reshaped_ref(:) - A_reshaped(:)));
if (diff < 0.001)
    disp('Reshape row-major test 2 worked (CPU)');
    fprintf('Processing time (trf+exec): %f\n',execTime+trfTime);
else
    error('Reshape row-major test 2 failed (CPU)');
end

%% Small matrix test 2 (CPU)
A = single([1 2 3; 4 5 6]);
desired_shape = [3 2 1 1];
[A_reshaped,execTime,trfTime] = mex_reshape_row_major(A,desired_shape);
A_reshaped_ref = reshape_row_major(A, desired_shape);

% Compare results
diff = sum(abs(A_reshaped_ref(:) - A_reshaped(:)));
if (diff < 0.001)
    disp('Reshape row-major test 3 worked (CPU)');
    fprintf('Processing time (trf+exec): %f\n',execTime+trfTime);
else
    error('Reshape row-major test 3 failed (CPU)');
end

%% Small matrix test 3 (CPU)
A = single([1 2 3; 4 5 6]);
desired_shape = [1 1 1 6];
[A_reshaped,execTime,trfTime] = mex_reshape_row_major(A,desired_shape);
A_reshaped_ref = reshape_row_major(A, desired_shape);

% Compare results
diff = sum(abs(A_reshaped_ref(:) - A_reshaped(:)));
if (diff < 0.001)
    disp('Reshape row-major test 3 worked (CPU)');
    fprintf('Processing time (trf+exec): %f\n',execTime+trfTime);
else
    error('Reshape row-major test 3 failed (CPU)');
end

%% 3d Matrix Test (CPU)
A = [1 2; 3 4];
A(:,:,2) = [5 6; 7 8];
A(:,:,3) = [9 10; 11 12]';
A = single(A);
desired_shape = [2 6 1 1];
[A_reshaped,execTime,trfTime] = mex_reshape_row_major(A,desired_shape);
A_reshaped_ref = reshape_row_major(A, desired_shape);

% Compare results
diff = sum(abs(A_reshaped_ref(:) - A_reshaped(:)));
if (diff < 0.001)
    disp('Reshape row-major test 3d worked (CPU)');
    fprintf('Processing time (trf+exec): %f\n',execTime+trfTime);
else
    error('Reshape row-major test 3d failed (CPU)');
end

%% 4d Matrix Test (CPU)
A = [1 2; 3 4];
A(:,:,2,1) = [5 6; 7 8];
A(:,:,3,1) = [9 10; 11 12];
A(:,:,1,2) = [13 14; 15 16];
A(:,:,2,2) = [17 18; 19 20];
A(:,:,3,2) = [21 22; 23 24];
A = single(A);
desired_shape = [2 12 1 1];
[A_reshaped,execTime,trfTime] = mex_reshape_row_major(A,desired_shape);
A_reshaped_ref = reshape_row_major(A, desired_shape);

% Compare results
diff = sum(abs(A_reshaped_ref(:) - A_reshaped(:)));
if (diff < 0.001)
    disp('Reshape row-major test 4d worked (CPU)');
    fprintf('Processing time (trf+exec): %f\n',execTime+trfTime);
else
    disp(A_reshaped);
    error('Reshape row-major test 4d failed (CPU)');
end

