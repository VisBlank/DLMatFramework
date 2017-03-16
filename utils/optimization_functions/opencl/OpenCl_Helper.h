#include <alloca.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef APPLE
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif


#define MAX_SOURCE_SIZE (0x100000)

int initializeOpenCl(cl_platform_id *cpPlatform, cl_device_id *device_id, cl_context *context, cl_command_queue *queue);
void shutdownOpenCl(cl_context context, cl_command_queue queue, cl_program program, cl_kernel *kernelList, int numKernel);
int compileOpenClSource(const char *kernelSource, cl_program *program, cl_device_id device_id, cl_context context);
int loadKernel(cl_program program, const char *kernel_name, cl_kernel *kernel);
const char *getErrorString(cl_int error);
