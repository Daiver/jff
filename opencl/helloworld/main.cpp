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

    //std::ifstream kernelFile("./kernel.cl");
    std::ifstream kernelFile("isrowscontainsamecolumn.cl");
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

    std::vector<float> a_data = {1, 2, 3, 4, 5, 6, 7};
    std::vector<float> b_data = {1, 2, 0, 0, 9, 0, 0};
    std::vector<float> c_data(a_data.size());
    const unsigned int count = a_data.size();

    cl_mem a_in = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * count, NULL, NULL);
    cl_mem b_in = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * count, NULL, NULL);
    cl_mem c_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * count, NULL, NULL);

    cl_kernel kernel = clCreateKernel(program, "vadd", &error);
    checkError (error);

    error  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_in);
    error |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_in);
    error |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &c_out);
    //error |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &count);
    checkError (error);

    error = clEnqueueWriteBuffer(commands, a_in, CL_FALSE, 0,
        sizeof(float) * count, a_data.data(), 0, NULL, NULL);
    error = clEnqueueWriteBuffer(commands, b_in, CL_FALSE, 0,
        sizeof(float) * count, b_data.data(), 0, NULL, NULL);
    checkError (error);

    const size_t global = count;
    const size_t local  = 1;
    error = clEnqueueNDRangeKernel(commands, kernel, 1, NULL,
            &global, &local, 0, NULL, NULL);
    checkError (error);

    error = clEnqueueReadBuffer( commands, c_out, CL_TRUE, 0,
            sizeof(float) * count, c_data.data(), 0, NULL, NULL ); 
    checkError (error);

    std::cout << "result:" << std::endl;
    for(int i = 0; i < c_data.size(); ++i)
        std::cout << c_data[i] << " ";
    std::cout << std::endl;

    std::cout << "End" << std::endl;
    return 0;
}
