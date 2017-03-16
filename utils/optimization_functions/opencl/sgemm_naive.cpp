/*
  Compiling on Linux
  g++ -I /usr/local/cuda-7.5/include/ -L /usr/local/cuda-7.5/lib64 -o
  matrix_big_mul_CL cl_mat_mult_2d.c -lOpenCL -lm
*/
#include <alloca.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "OpenCl_Helper.h"
#define DIMS_WORK_GROUP 1
/*#define KERNEL_FILENAME "./optimization_functions/opencl/matrix_mul_coarsening.cl"*/
/*#define KERNEL_FILENAME "./optimization_functions/opencl/matrix_mul_data_reuse.cl"*/
/*#define KERNEL_FILENAME "./optimization_functions/opencl/matrix_mul_data_local.cl"*/
#define KERNEL_FILENAME "./utils/optimization_functions/opencl/sgemm_naive.cl"

cl_platform_id cpPlatform;     /* OpenCL platform */
cl_device_id device_id;        /* device ID */
cl_context context;            /* context */
cl_command_queue queue;        /* command queue */
cl_program program;            /* program */
cl_kernel kernel;              /* kernel */

/* Our contract */
void matrix_2d_mul_float(float *A, float *B, float *biasVec, float *C, int num_rows_A, int num_cols_A, int num_rows_B, int num_cols_B , int biasVecSize, double *execTime, double *trfTime) {
  /* Define pointers to device memory */
  cl_mem d_a;
  cl_mem d_b;
  cl_mem d_bias_vec;
  cl_mem d_c;

  int result;
  printf("Test OpenCl matrix multiplication\n");
  result = initializeOpenCl(&cpPlatform, &device_id, &context, &queue);
  if (result != 1) {
    printf("OpenCl load error abort\n");
    return;
  }

  result = compileOpenClSource(KERNEL_FILENAME, &program, device_id, context);

  if (result != 1) {
    printf("compileKernel failed abort\n");
    return;
  }
  result = loadKernel(program, "matrix_2d_mul_float_gpu", &kernel);
  if (result != 1) {
    printf("loadKernel failed abort\n");
    return;
  }

  cl_int err;
  cl_ulong time_start, time_end; /* Time */
  double kernel_exec_time;

  int numBytesA = sizeof(float) * num_rows_A * num_cols_A;
  int numBytesB = sizeof(float) * num_rows_B * num_cols_B;
  int numBytesBiasVec = sizeof(float) * biasVecSize;
  int numBytesC = sizeof(float) * num_rows_A * num_cols_B;

  printf("A[%d, %d] x B[%d, %d] = C[%d, %d]\n", num_rows_A,num_cols_A,num_rows_B,num_cols_B,num_rows_A,num_cols_B);
  /* Print matrices */
  /*printf("\nMatrix A:\n");
  int idxA = 0;
  for (idxA = 0; idxA < (num_rows_A*num_cols_A); idxA++) {
    printf("A[%d]=%.2f\n",idxA, A[idxA]);
  }

  printf("\nMatrix B:\n");
  int idxB = 0;
  for (idxB = 0; idxB < (num_rows_B*num_cols_B); idxB++) {
    printf("B[%d]=%.2f\n",idxB, B[idxB]);
  }

  printf("\nMatrix C:\n");
  for (idxB = 0; idxB < biasVecSize; idxB++) {
    printf("C[%d]=%.2f\n",idxB, biasVec[idxB]);
  }*/

  /* Measure elapsed wall time */
  struct timespec now, tmstart;
  /*tic*/
  clock_gettime(CLOCK_REALTIME, &tmstart);

  /*
   * Create the input and output arrays in device memory for our calculation
  */
  d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, numBytesA, NULL, NULL);
  d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, numBytesB, NULL, NULL);
  d_bias_vec = clCreateBuffer(context, CL_MEM_READ_ONLY, numBytesBiasVec, NULL, NULL);
  d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, numBytesC, NULL, NULL);

  /* Configure the global size and local size. */
  /*size_t localSize[DIMS_WORK_GROUP], globalSize[DIMS_WORK_GROUP];
  /* number of work-items that make up a work-group */
  /*localSize[0] = num_rows_A/2;
  /*localSize[1] = N_THREADS;*/
  /*globalSize[x] must be divisible by localSize[x]*/

  /*globalSize[0] = ceil(num_rows_A / (float)N_THREADS) * N_THREADS;
  globalSize[1] = ceil(num_cols_B / (float)N_THREADS) * N_THREADS;*/

  size_t globalSize[2];
  globalSize[0] = num_rows_A;
  globalSize[1] = num_cols_B;

  printf("\nGlobal size[%d, %d]\n", (unsigned int)globalSize[0],(unsigned int)globalSize[1]);
  /*printf("\nGlobal size[%d] LocalSize[%d]\n", (unsigned int)globalSize[0],(unsigned int)localSize[0]);*/

  /* Copy matrices A and B to GPU */
  clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0, numBytesA, A, 0, NULL, NULL);
  clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0, numBytesB, B, 0, NULL, NULL);
  clEnqueueWriteBuffer(queue, d_bias_vec, CL_TRUE, 0, numBytesBiasVec, biasVec, 0, NULL, NULL);

  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_a);
  err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_b);
  err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_bias_vec);
  err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&d_c);
  err |= clSetKernelArg(kernel, 4, sizeof(int), (void *)&num_rows_A);
  err |= clSetKernelArg(kernel, 5, sizeof(int), (void *)&num_cols_A);
  err |= clSetKernelArg(kernel, 6, sizeof(int), (void *)&num_rows_B);
  err |= clSetKernelArg(kernel, 7, sizeof(int), (void *)&num_cols_B);

  if (err != CL_SUCCESS) {
    printf("Error: Failed to set kernel arguments! %d <%s>\n", err, getErrorString(err));
    return;
  }

  /* Enqueues a command to execute a kernel on a device. (2 dimensions) */
  cl_event kernEvent;
  /*err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, localSize,*/
  /* Changed to automatically infer the number of threads */
  /*err = clEnqueueNDRangeKernel(queue, kernel, DIMS_WORK_GROUP, NULL, globalSize, localSize,*/
  err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, NULL,
    0, NULL, &kernEvent);
    if (err != CL_SUCCESS) {
      printf("Failed to launch kernel! %d <%s>\n", err, getErrorString(err));
      return;
    }
    /* Wait for the command queue to get serviced before reading back results */
    clFinish(queue);

    /* Get the result from the GPU to the CPU */
    clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0, numBytesC, C, 0, NULL, NULL);

    clGetEventProfilingInfo(kernEvent, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(kernEvent, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    kernel_exec_time = time_end - time_start;
    kernel_exec_time /= 1000000000;

    /*toc*/
    clock_gettime(CLOCK_REALTIME, &now);
    double wall_time_sec = (double)((now.tv_sec+now.tv_nsec*1e-9) - (double)(tmstart.tv_sec+tmstart.tv_nsec*1e-9));
    *execTime = kernel_exec_time;
    *trfTime = wall_time_sec - kernel_exec_time;

    /* Release memories from GPU */
    clReleaseMemObject(d_a);
    clReleaseMemObject(d_bias_vec);
    clReleaseMemObject(d_b);
    clReleaseMemObject(d_c);

    shutdownOpenCl(context, queue, program, &kernel, 1);
  }
