#include <utility>
#include "CL/cl.h"

#include <cstdio>
#include <vector>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <iterator>
#include <streambuf>

std::string desctiptionForError(const cl_int error)
{
    switch (error) {
    case CL_BUILD_PROGRAM_FAILURE:
        return std::string("CL_BUILD_PROGRAM_FAILURE");
        break;
    case CL_COMPILER_NOT_AVAILABLE:
        return std::string("CL_COMPILER_NOT_AVAILABLE");
        break;
    case CL_DEVICE_NOT_AVAILABLE:
        return std::string("CL_DEVICE_NOT_AVAILABLE");
        break;
    case CL_DEVICE_NOT_FOUND:
        return std::string("CL_DEVICE_NOT_FOUND");
        break;
    case CL_IMAGE_FORMAT_MISMATCH:
        return std::string("CL_IMAGE_FORMAT_MISMATCH");
        break;
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:
        return std::string("CL_IMAGE_FORMAT_NOT_SUPPORTED");
        break;
    case CL_INVALID_ARG_INDEX:
        return std::string("CL_INVALID_ARG_INDEX");
        break;
    case CL_INVALID_ARG_SIZE:
        return std::string("CL_INVALID_ARG_SIZE");
        break;
    case CL_INVALID_ARG_VALUE:
        return std::string("CL_INVALID_ARG_VALUE");
        break;
    case CL_INVALID_BINARY:
        return std::string("CL_INVALID_BINARY");
        break;
    case CL_INVALID_BUFFER_SIZE:
        return std::string("CL_INVALID_BUFFER_SIZE");
        break;
    case CL_INVALID_BUILD_OPTIONS:
        return std::string("CL_INVALID_BUILD_OPTIONS");
        break;
    case CL_INVALID_COMMAND_QUEUE:
        return std::string("CL_INVALID_COMMAND_QUEUE");
        break;
    case CL_INVALID_CONTEXT:
        return std::string("CL_INVALID_CONTEXT");
        break;
    case CL_INVALID_DEVICE:
        return std::string("CL_INVALID_DEVICE");
        break;
    case CL_INVALID_DEVICE_TYPE:
        return std::string("CL_INVALID_DEVICE_TYPE");
        break;
    case CL_INVALID_EVENT:
        return std::string("CL_INVALID_EVENT");
        break;
    case CL_INVALID_EVENT_WAIT_LIST:
        return std::string("CL_INVALID_EVENT_WAIT_LIST");
        break;
    case CL_INVALID_GL_OBJECT:
        return std::string("CL_INVALID_GL_OBJECT");
        break;
    case CL_INVALID_GLOBAL_OFFSET:
        return std::string("CL_INVALID_GLOBAL_OFFSET");
        break;
    case CL_INVALID_HOST_PTR:
        return std::string("CL_INVALID_HOST_PTR");
        break;
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
        return std::string("CL_INVALID_IMAGE_FORMAT_DESCRIPTOR");
        break;
    case CL_INVALID_IMAGE_SIZE:
        return std::string("CL_INVALID_IMAGE_SIZE");
        break;
    case CL_INVALID_KERNEL_NAME:
        return std::string("CL_INVALID_KERNEL_NAME");
        break;
    case CL_INVALID_KERNEL:
        return std::string("CL_INVALID_KERNEL");
        break;
    case CL_INVALID_KERNEL_ARGS:
        return std::string("CL_INVALID_KERNEL_ARGS");
        break;
    case CL_INVALID_KERNEL_DEFINITION:
        return std::string("CL_INVALID_KERNEL_DEFINITION");
        break;
    case CL_INVALID_MEM_OBJECT:
        return std::string("CL_INVALID_MEM_OBJECT");
        break;
    case CL_INVALID_OPERATION:
        return std::string("CL_INVALID_OPERATION");
        break;
    case CL_INVALID_PLATFORM:
        return std::string("CL_INVALID_PLATFORM");
        break;
    case CL_INVALID_PROGRAM:
        return std::string("CL_INVALID_PROGRAM");
        break;
    case CL_INVALID_PROGRAM_EXECUTABLE:
        return std::string("CL_INVALID_PROGRAM_EXECUTABLE");
        break;
    case CL_INVALID_QUEUE_PROPERTIES:
        return std::string("CL_INVALID_QUEUE_PROPERTIES");
        break;
    case CL_INVALID_SAMPLER:
        return std::string("CL_INVALID_SAMPLER");
        break;
    case CL_INVALID_VALUE:
        return std::string("CL_INVALID_VALUE");
        break;
    case CL_INVALID_WORK_DIMENSION:
        return std::string("CL_INVALID_WORK_DIMENSION");
        break;
    case CL_INVALID_WORK_GROUP_SIZE:
        return std::string("CL_INVALID_WORK_GROUP_SIZE");
        break;
    case CL_INVALID_WORK_ITEM_SIZE:
        return std::string("CL_INVALID_WORK_ITEM_SIZE");
        break;
    case CL_MAP_FAILURE:
        return std::string("CL_MAP_FAILURE");
        break;
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
        return std::string("CL_MEM_OBJECT_ALLOCATION_FAILURE");
        break;
    case CL_MEM_COPY_OVERLAP:
        return std::string("CL_MEM_COPY_OVERLAP");
        break;
    case CL_OUT_OF_HOST_MEMORY:
        return std::string("CL_OUT_OF_HOST_MEMORY");
        break;
    case CL_OUT_OF_RESOURCES:
        return std::string("CL_OUT_OF_RESOURCES");
        break;
    case CL_PROFILING_INFO_NOT_AVAILABLE:
        return std::string("CL_PROFILING_INFO_NOT_AVAILABLE");
        break;

    default:
        break;
    }
    return std::string("UNKNOWN ERROR CODE");
}

std::string GetPlatformName (cl_platform_id id)
{
    size_t size = 0;
    clGetPlatformInfo (id, CL_PLATFORM_NAME, 0, nullptr, &size);

    std::string result;
    result.resize (size);
    clGetPlatformInfo (id, CL_PLATFORM_NAME, size,
        const_cast<char*> (result.data ()), nullptr);

    return result;
}

std::string GetDeviceName (cl_device_id id)
{
    size_t size = 0;
    clGetDeviceInfo (id, CL_DEVICE_NAME, 0, nullptr, &size);

    std::string result;
    result.resize (size);
    clGetDeviceInfo (id, CL_DEVICE_NAME, size,
        const_cast<char*> (result.data ()), nullptr);

    return result;
}


void checkError (cl_int error)
{
    if (error != CL_SUCCESS) {
        std::cerr << "OpenCL call failed with error " << error << std::endl;
		std::cerr << desctiptionForError(error) << std::endl;
        std::exit (1);
    }
}

void printBuildProgramInfo(const cl_program &program, const cl_device_id device_id)
{
    size_t len; char buffer[2048];
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer),
            buffer, &len);
    printf("%s\n", buffer);
}

int main()
{

    cl_uint platformIdCount = 0;
    clGetPlatformIDs (0, nullptr, &platformIdCount);

    if (platformIdCount == 0) {
        std::cerr << "No OpenCL platform found" << std::endl;
        return 1;
    } else {
        std::cout << "Found " << platformIdCount << " platform(s)" << std::endl;
    }

    std::vector<cl_platform_id> platformIds (platformIdCount);
    clGetPlatformIDs (platformIdCount, platformIds.data (), nullptr);

    for (cl_uint i = 0; i < platformIdCount; ++i) {
        std::cout << "\t (" << (i+1) << ") : " << GetPlatformName (platformIds [i]) << std::endl;
    }

    cl_uint deviceIdCount = 0;
    const int platformInd = 1;
    clGetDeviceIDs (platformIds[platformInd], CL_DEVICE_TYPE_ALL, 0, nullptr,
        &deviceIdCount);

    if (deviceIdCount == 0) {
        std::cerr << "No OpenCL devices found" << std::endl;
        return 1;
    } else {
        std::cout << "Found " << deviceIdCount << " device(s)" << std::endl;
    }

    std::vector<cl_device_id> deviceIds (deviceIdCount);
    clGetDeviceIDs (platformIds[platformInd], CL_DEVICE_TYPE_ALL, deviceIdCount,
        deviceIds.data (), nullptr);

    for (cl_uint i = 0; i < deviceIdCount; ++i) {
        std::cout << "\t (" << (i+1) << ") : " << GetDeviceName (deviceIds [i]) << std::endl;
    }

    const cl_context_properties contextProperties [] =
    {
        CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties> (platformIds[platformInd]),
        0, 0
    };

    cl_int error = CL_SUCCESS;
    cl_context context = clCreateContext (contextProperties, deviceIdCount,
        deviceIds.data (), nullptr, nullptr, &error);
    checkError (error);

    std::cout << "Context created" << std::endl;

    cl_command_queue commands = clCreateCommandQueue(context, deviceIds[0], 0, &error);
    //cl_command_queue commands = clCreateCommandQueueWithProperties(context, deviceIds[0], 0, &error);
    checkError (error);

    std::ifstream kernelFile("./kernel.cl");
    //std::ifstream kernelFile("./kernel2.cl");
    std::string kernelSource((std::istreambuf_iterator<char>(kernelFile)),
                             std::istreambuf_iterator<char>());

    const char *cKernelSource = kernelSource.c_str();
    cl_program program = clCreateProgramWithSource(context, 1,
        (const char **) &cKernelSource, NULL, &error);
    checkError (error);

    error = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    printBuildProgramInfo(program, deviceIds[0]);
    checkError (error);

    const unsigned int count = 1;

    std::vector<int> a = {0};
    
    cl_mem a_in = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * count, NULL, NULL);

    //cl_kernel kernel = clCreateKernel(program, "atomicSum", &error);
    //cl_kernel kernel = clCreateKernel(program, "atomicSum2", &error);
    cl_kernel kernel = clCreateKernel(program, "atomicSum3", &error);
    checkError (error);

    error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_in);
    checkError (error);

    error = clEnqueueWriteBuffer(commands, a_in, CL_FALSE, 0,
        sizeof(int) * count, a.data(), 0, NULL, NULL);
    checkError (error);

    const size_t countOfSum = 100000000;
    const size_t global = countOfSum;
    const size_t local  = 100;
    error = clEnqueueNDRangeKernel(commands, kernel, 1, NULL,
            &global, &local, 0, NULL, NULL);
    checkError (error);

    error = clEnqueueReadBuffer( commands, a_in, CL_TRUE, 0,
            sizeof(int) * count, a.data(), 0, NULL, NULL ); 
    checkError (error);

    std::cout << "result:" << std::endl;
    for(int i = 0; i < a.size(); ++i)
        std::cout << a[i] << " ";
    std::cout << std::endl;

    std::cout << "End" << std::endl;
    return 0;
}
