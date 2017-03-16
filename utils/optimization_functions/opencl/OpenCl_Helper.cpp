#include "OpenCl_Helper.h"

int initializeOpenCl(cl_platform_id *cpPlatform, cl_device_id *device_id, cl_context *context, cl_command_queue *queue) {
  cl_int err;
  printf("Initializing OpenCL device...\n");

  /* Obtain the list of platforms available. Get the ID for the first one */
  err = clGetPlatformIDs(1, cpPlatform, NULL);
  if (err != CL_SUCCESS) {
    printf("Error: Failed to get Platform ID! %s\n", getErrorString(err));
    return - 1;
  }

  /* Ask for 1 GPU */
  err = clGetDeviceIDs(*cpPlatform, CL_DEVICE_TYPE_GPU, 1, device_id, NULL);
  if (err != CL_SUCCESS) {
    printf("Error: Failed to get GPU! %s\n", getErrorString(err));
    return - 1;
  }

  /* Create a context */
  *context = clCreateContext(0, 1, device_id, NULL, NULL, &err);
  if (err != CL_SUCCESS) {
    printf("Error: Failed to create context! %s\n", getErrorString(err));
    return - 1;
  }
  /* Create a command queue */
  /* queue = clCreateCommandQueue(context, device_id, 0, &err); */
  /* Enable queue profile */
  *queue =
  clCreateCommandQueue(*context, *device_id, CL_QUEUE_PROFILING_ENABLE, &err);
  if (!(*queue)) {
    printf("Failed to create command queue. Error %s\n",getErrorString(err));
    return -1;
  }
  return 1;
}

void shutdownOpenCl(cl_context context, cl_command_queue queue, cl_program program, cl_kernel *kernelList, int numKernel) {
  int cont_kernel;
  cl_int err;
  printf("Releasing context.\n");
  err = clReleaseContext(context);
  if (err != CL_SUCCESS) {
    printf("Error: Failed to release context %s\n", getErrorString(err));
  }

  for (cont_kernel = 0; cont_kernel < numKernel; cont_kernel++) {
    printf("Releasing kernel %d.\n",cont_kernel);
    err = clReleaseKernel(kernelList[cont_kernel]);
    if (err != CL_SUCCESS) {
      printf("Error: Failed to release kernel %s\n", getErrorString(err));
    }
  }

  printf("Releasing program.\n");
  err = clReleaseProgram(program);
  if (err != CL_SUCCESS) {
    printf("Error: Failed to release program %s\n", getErrorString(err));
  }

  printf("Releasing queue.\n");
  err = clReleaseCommandQueue(queue);
  if (err != CL_SUCCESS) {
    printf("Error: Failed to release queue %s\n", getErrorString(err));
  }
}

int loadKernel(cl_program program, const char *kernel_name, cl_kernel *kernel) {
  cl_int err;
  /* Create the compute kernel in the program we wish to run */
  *kernel = clCreateKernel(program, kernel_name, &err);
  if (err != CL_SUCCESS) {
    printf("Error: Failed to create kernel %s\n", getErrorString(err));
    return -1;
  }
  return 1;
}

int compileOpenClSource(const char *kernelSource, cl_program *program, cl_device_id device_id, cl_context context) {
  printf("Loading kernel file...\n");
  /* Load the source code containing the kernel */
  FILE *fp;
  char *source_str;
  size_t source_size;
  fp = fopen(kernelSource, "r");
  if (!fp) {
    fprintf(stderr, "Failed to load kernel: %s\n",kernelSource);
    return -1;
  }
  /* Read our kernel to memory (source_str) */
  source_str = (char *)malloc(MAX_SOURCE_SIZE);
  source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
  fclose(fp);

  cl_int err;
  /* Create the compute program from the source buffer */
  *program = clCreateProgramWithSource(context, 1, (const char **)&source_str,
  (const size_t *)&source_size, &err);
  free(source_str);

  /* Build the program (with our kernel) */
  printf("Compiling OpenCL kernel...\n");
  const char options[] = "-cl-std=CL1.2";
  err = clBuildProgram(*program, 0, NULL, options, NULL, NULL);
  if (err != CL_SUCCESS) {
    cl_build_status status;
    char *programLog;
    size_t logSize;
    printf("Kernel compilation error\n");
    clGetProgramBuildInfo(*program, device_id, CL_PROGRAM_BUILD_STATUS,sizeof(cl_build_status), &status, NULL);
    /* check build log */
    clGetProgramBuildInfo(*program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL,&logSize);
    programLog = (char *)calloc(logSize + 1, sizeof(char));
    clGetProgramBuildInfo(*program, device_id, CL_PROGRAM_BUILD_LOG, logSize + 1,programLog, NULL);
    programLog[logSize] = '\0';
    printf("Build failed; error=%d, status=%d, programLog:nn%s\n", err, status,programLog);
    free(programLog);
    return -1;
  }

  return 1;
}

const char *getErrorString(cl_int error)
{
  switch(error){
    /* run-time and JIT compiler errors */
    case 0: return "CL_SUCCESS";
    case -1: return "CL_DEVICE_NOT_FOUND";
    case -2: return "CL_DEVICE_NOT_AVAILABLE";
    case -3: return "CL_COMPILER_NOT_AVAILABLE";
    case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case -5: return "CL_OUT_OF_RESOURCES";
    case -6: return "CL_OUT_OF_HOST_MEMORY";
    case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case -8: return "CL_MEM_COPY_OVERLAP";
    case -9: return "CL_IMAGE_FORMAT_MISMATCH";
    case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case -11: return "CL_BUILD_PROGRAM_FAILURE";
    case -12: return "CL_MAP_FAILURE";
    case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case -15: return "CL_COMPILE_PROGRAM_FAILURE";
    case -16: return "CL_LINKER_NOT_AVAILABLE";
    case -17: return "CL_LINK_PROGRAM_FAILURE";
    case -18: return "CL_DEVICE_PARTITION_FAILED";
    case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

    /* compile-time errors */
    case -30: return "CL_INVALID_VALUE";
    case -31: return "CL_INVALID_DEVICE_TYPE";
    case -32: return "CL_INVALID_PLATFORM";
    case -33: return "CL_INVALID_DEVICE";
    case -34: return "CL_INVALID_CONTEXT";
    case -35: return "CL_INVALID_QUEUE_PROPERTIES";
    case -36: return "CL_INVALID_COMMAND_QUEUE";
    case -37: return "CL_INVALID_HOST_PTR";
    case -38: return "CL_INVALID_MEM_OBJECT";
    case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case -40: return "CL_INVALID_IMAGE_SIZE";
    case -41: return "CL_INVALID_SAMPLER";
    case -42: return "CL_INVALID_BINARY";
    case -43: return "CL_INVALID_BUILD_OPTIONS";
    case -44: return "CL_INVALID_PROGRAM";
    case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case -46: return "CL_INVALID_KERNEL_NAME";
    case -47: return "CL_INVALID_KERNEL_DEFINITION";
    case -48: return "CL_INVALID_KERNEL";
    case -49: return "CL_INVALID_ARG_INDEX";
    case -50: return "CL_INVALID_ARG_VALUE";
    case -51: return "CL_INVALID_ARG_SIZE";
    case -52: return "CL_INVALID_KERNEL_ARGS";
    case -53: return "CL_INVALID_WORK_DIMENSION";
    case -54: return "CL_INVALID_WORK_GROUP_SIZE";
    case -55: return "CL_INVALID_WORK_ITEM_SIZE";
    case -56: return "CL_INVALID_GLOBAL_OFFSET";
    case -57: return "CL_INVALID_EVENT_WAIT_LIST";
    case -58: return "CL_INVALID_EVENT";
    case -59: return "CL_INVALID_OPERATION";
    case -60: return "CL_INVALID_GL_OBJECT";
    case -61: return "CL_INVALID_BUFFER_SIZE";
    case -62: return "CL_INVALID_MIP_LEVEL";
    case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
    case -64: return "CL_INVALID_PROPERTY";
    case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
    case -66: return "CL_INVALID_COMPILER_OPTIONS";
    case -67: return "CL_INVALID_LINKER_OPTIONS";
    case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

    /* extension errors */
    case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
    case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
    case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
    case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
    case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
    case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
    default: return "Unknown OpenCL error";
  }
}
