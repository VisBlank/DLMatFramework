%% Create Data
%A = single([1 2; 3 4]);
%B = single([5 6; 7 8]);
A = single(randn(1000,500));
B = single(randn(500,1000));
tic;result_sgemm_ref = A*B;time_matlab = toc;
dimA = size(A);
dimB = size(B);

%% Test SGEMM on OpenCL
[result, execTime, trfTime] = mex_matMult2D_CL(A,B,single(zeros(dimA(1),1)));
timeTotal = sum([execTime trfTime]);
diff = sum(abs(result_sgemm_ref(:) - result(:)));
if (diff < 10)
    disp('small SGEMM OpenCl test worked');
    fprintf('Diff: %f, Processing time (trf+exec): %f matlab time(tic..toc): %f\n',diff, timeTotal,time_matlab);
else
    fprintf('Error: %f\n',diff);
    error('small SGEMM OpenCl test failed');
end