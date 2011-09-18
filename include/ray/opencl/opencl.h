#ifndef _RAY_OPENCL_OPENCL_H_
#define _RAY_OPENCL_OPENCL_H_

#if defined (__APPLE__)
#  include <OpenCL/cl.h>
#else
#  include <CL/cl.h>
#endif


#include <ray/opencl/OCLConstants.h>
#include <ray/opencl/OCLError.h>
#include <ray/opencl/OCLUtils.h>
#include <ray/opencl/ocl_device_properties.h>
#include <ray/opencl/OCLDevice.h>
#include <ray/opencl/OCLPlatform.h>
#include <ray/opencl/OCLContext.h>
#include <ray/opencl/OCLProgram.h>
#include <ray/opencl/OCLBuffer.h>
#include <ray/opencl/OCLEvent.h>
#include <ray/opencl/OCLKernel.h>
#include <ray/opencl/OCLCommandQueue.h>


#pragma comment(lib, "OpenCL")


#endif