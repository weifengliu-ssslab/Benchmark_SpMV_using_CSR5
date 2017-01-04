#ifndef BASICCL_H
#define BASICCL_H

#define CL_STRING_LENGTH 128

class BasicCL
{
public:
    BasicCL();

    int getNumPlatform(cl_uint *numPlatforms);
    int getPlatformIDs(cl_platform_id *platforms, cl_uint numPlatforms);
    int getPlatformInfo(cl_platform_id platform, char *platformVendor, char *platformVersion);

    int getNumCpuDevices(cl_platform_id platform, cl_uint *numCpuDevices);
    int getNumGpuDevices(cl_platform_id platform, cl_uint *numGpuDevices);
    int getCpuDeviceIDs(cl_platform_id platform, cl_uint numCpuDevices, cl_device_id *cpuDevices);
    int getGpuDeviceIDs(cl_platform_id platform, cl_uint numGpuDevices, cl_device_id *gpuDevices);
    int getDeviceInfo(cl_device_id device, char *deviceName, char *deviceVersion,
                      int *deviceComputeUnits, cl_ulong *deviceGlobalMem,
                      cl_ulong *deviceLocalMem, int *maxSubDevices);

    int getContext(cl_context *context, cl_device_id *devices, cl_uint numDevices);

    int getCommandQueue(cl_command_queue *commandQueue, cl_context context, cl_device_id device);
    int getCommandQueueProfilingEnable(cl_command_queue *commandQueue, cl_context context, cl_device_id device);

    int getProgram(cl_program *program, cl_context context, const char *kernelSourceCode);
    int getProgramFromFile(cl_program *program, cl_context context, const char *sourceFilename);
    int getKernel(cl_kernel *kernel, cl_program program, const char *kernelName);
    int getEventTimer(cl_event event,
                      cl_ulong *queuedTime, cl_ulong *submitTime, 
                      cl_ulong *startTime,  cl_ulong *endTime);

private:
    cl_int  _ciErr;
};

BasicCL::BasicCL()
{
}

int BasicCL::getNumPlatform(cl_uint *numPlatforms)
{
    _ciErr = clGetPlatformIDs(0, NULL, numPlatforms);
    return _ciErr;
}

int BasicCL::getPlatformIDs(cl_platform_id *platforms, cl_uint numPlatforms)
{
    _ciErr = clGetPlatformIDs(numPlatforms, platforms, NULL);
    return _ciErr;
}

int BasicCL::getPlatformInfo(cl_platform_id platform, char *platformVendor, char *platformVersion)
{
    _ciErr = clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, CL_STRING_LENGTH*sizeof(char), platformVendor, NULL);
    if(_ciErr != CL_SUCCESS) return _ciErr;
    _ciErr = clGetPlatformInfo(platform, CL_PLATFORM_VERSION, CL_STRING_LENGTH*sizeof(char), platformVersion, NULL);
    if(_ciErr != CL_SUCCESS) return _ciErr;
    return CL_SUCCESS;
}

int BasicCL::getNumCpuDevices(cl_platform_id platform, cl_uint *numCpuDevices)
{
    _ciErr = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, NULL, numCpuDevices);
    return _ciErr;
}

int BasicCL::getCpuDeviceIDs(cl_platform_id platform, cl_uint numCpuDevices, cl_device_id *CpuDevices)
{
    _ciErr = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, numCpuDevices, CpuDevices, NULL);
    return _ciErr;
}

int BasicCL::getNumGpuDevices(cl_platform_id platform, cl_uint *numGpuDevices)
{
    _ciErr = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, numGpuDevices);
    return _ciErr;
}

int BasicCL::getGpuDeviceIDs(cl_platform_id platform, cl_uint numGpuDevices, cl_device_id *GpuDevices)
{
    _ciErr = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numGpuDevices, GpuDevices, NULL);
    return _ciErr;
}

int BasicCL::getDeviceInfo(cl_device_id device, char *deviceName, char *deviceVersion,
                           int *deviceComputeUnits, cl_ulong *deviceGlobalMem,
                           cl_ulong *deviceLocalMem, int *maxSubDevices)
{
    _ciErr = clGetDeviceInfo(device, CL_DEVICE_NAME, CL_STRING_LENGTH*sizeof(char), deviceName, NULL);
    if(_ciErr != CL_SUCCESS) return _ciErr;
    _ciErr = clGetDeviceInfo(device, CL_DEVICE_VERSION, CL_STRING_LENGTH*sizeof(char), deviceVersion, NULL);
    if(_ciErr != CL_SUCCESS) return _ciErr;
    _ciErr = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), deviceComputeUnits, NULL);
    if(_ciErr != CL_SUCCESS) return _ciErr;
    _ciErr = clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), deviceGlobalMem, NULL);
    if(_ciErr != CL_SUCCESS) return _ciErr;
    _ciErr = clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), deviceLocalMem, NULL);
    if(_ciErr != CL_SUCCESS) return _ciErr;
    //_ciErr = clGetDeviceInfo(device, CL_DEVICE_PARTITION_MAX_SUB_DEVICES, sizeof(cl_uint), maxSubDevices, NULL);
    //if(_ciErr != CL_SUCCESS) return _ciErr;
    return CL_SUCCESS;
}

int BasicCL::getContext(cl_context *context, cl_device_id *devices, cl_uint numDevices)
{
    context[0] = clCreateContext(0, numDevices, devices, NULL, NULL, &_ciErr);
    //cout << "------------ _ciErr " << _ciErr << endl;
    return _ciErr;
}

int BasicCL::getCommandQueue(cl_command_queue *commandQueue, cl_context context, cl_device_id device)
{
    // use clCreateCommandQueueWithProperties for ocl2.0
    //commandQueue[0] = clCreateCommandQueue(context, device, NULL, &_ciErr);
    //CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
    //CL_QUEUE_PROFILING_ENABLE
    //cout << "------------ _ciErr " << _ciErr << endl;
    return _ciErr;
}

int BasicCL::getCommandQueueProfilingEnable(cl_command_queue *commandQueue, cl_context context, cl_device_id device)
{
    // use clCreateCommandQueueWithProperties for ocl2.0
    commandQueue[0] = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &_ciErr);
    //CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
    //CL_QUEUE_PROFILING_ENABLE
    //cout << "------------ _ciErr " << _ciErr << endl;
    return _ciErr;
}

int BasicCL::getProgram(cl_program *program, cl_context context, const char *kernelSourceCode)
{
    size_t SourceSize[] = { strlen(kernelSourceCode)};
    program[0] = clCreateProgramWithSource(context, 1, &kernelSourceCode, SourceSize, &_ciErr); //cout << "------------ _ciErr " << _ciErr << endl;
    if(_ciErr != CL_SUCCESS) return _ciErr;
    //_ciErr = clBuildProgram(program[0], 0, NULL, "-cl-opt-disable", NULL, NULL); //cout << "------------ _ciErr " << _ciErr << endl;
    _ciErr = clBuildProgram(program[0], 0, NULL, NULL, NULL, NULL); //cout << "------------ _ciErr " << _ciErr << endl;
    if(_ciErr != CL_SUCCESS) return _ciErr;
    return CL_SUCCESS;
}

char* readSource(const char *sourceFilename) {
    
    FILE *fp;
    int err;
    int size;
    
    char *source;
    
    fp = fopen(sourceFilename, "rb");
    if(fp == NULL) {
        printf("Could not open kernel file: %s\n", sourceFilename);
        exit(-1);
    }
    
    err = fseek(fp, 0, SEEK_END);
    if(err != 0) {
        printf("Error seeking to end of file\n");
        exit(-1);
    }
    
    size = ftell(fp);
    if(size < 0) {
        printf("Error getting file position\n");
        exit(-1);
    }
    
    err = fseek(fp, 0, SEEK_SET);
    if(err != 0) {
        printf("Error seeking to start of file\n");
        exit(-1);
    }
    
    source = (char*)malloc(size+1);
    if(source == NULL) {
        printf("Error allocating %d bytes for the program source\n", size+1);
        exit(-1);
    }
    
    err = fread(source, 1, size, fp);
    if(err != size) {
        printf("only read %d bytes\n", err);
        exit(0);
    }
    
    source[size] = '\0';
    
    fclose (fp);
    
    return source;
}

int BasicCL::getProgramFromFile(cl_program *program, cl_context context, const char *sourceFilename)
{
    char *kernelSourceCode;
    kernelSourceCode = readSource(sourceFilename);
    //cout << kernelSourceCode << endl;
    int err = getProgram(program, context, kernelSourceCode);
    return err;
}

int BasicCL::getKernel(cl_kernel *kernel, cl_program program, const char *kernelName)
{
    kernel[0] = clCreateKernel(program, kernelName, &_ciErr);
    return _ciErr;
}

int BasicCL::getEventTimer(cl_event event,
                           cl_ulong *queuedTime, cl_ulong *submitTime, 
                           cl_ulong *startTime,  cl_ulong *endTime)
{
    _ciErr = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), queuedTime, NULL);
    if(_ciErr != CL_SUCCESS) return _ciErr;
    
    _ciErr = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong), submitTime, NULL);
    if(_ciErr != CL_SUCCESS) return _ciErr;
    
    _ciErr = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,  sizeof(cl_ulong), startTime, NULL);
    if(_ciErr != CL_SUCCESS) return _ciErr;
    
    _ciErr = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,    sizeof(cl_ulong), endTime, NULL);
    if(_ciErr != CL_SUCCESS) return _ciErr;
    
    return CL_SUCCESS;
}

#endif // BASICCL_H
