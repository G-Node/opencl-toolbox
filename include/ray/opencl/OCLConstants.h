#ifndef _RAY_OPENCL_OCLCONSTANTS_H_
#define _RAY_OPENCL_OCLCONSTANTS_H_

/*
 * OpenCL constants packaged and redefined to consolidate information
 *
 * Author: Radford Juang 
 * Date:  4.27.2010
 * Email: rayver /_at_/ hkn (dot) berkeley (dot) edu
 */

#include <CL/cl.h>

namespace ray { namespace opencl {

typedef enum {
	DEVICE_TYPE_DEFAULT		= CL_DEVICE_TYPE_DEFAULT,
	DEVICE_TYPE_CPU			= CL_DEVICE_TYPE_CPU,
	DEVICE_TYPE_GPU			= CL_DEVICE_TYPE_GPU,
	DEVICE_TYPE_ACCELERATOR = CL_DEVICE_TYPE_ACCELERATOR,
	DEVICE_TYPE_ALL			= CL_DEVICE_TYPE_ALL	
} ocl_device_type;

typedef enum {
	MEM_FLAGS_READ_WRITE     = CL_MEM_READ_WRITE,
	MEM_FLAGS_WRITE_ONLY     = CL_MEM_WRITE_ONLY,
	MEM_FLAGS_READ_ONLY      = CL_MEM_READ_ONLY,
	MEM_FLAGS_USE_HOST_PTR   = CL_MEM_USE_HOST_PTR,
	MEM_FLAGS_ALLOC_HOST_PTR = CL_MEM_ALLOC_HOST_PTR,
	MEM_FLAGS_COPY_HOST_PTR  = CL_MEM_COPY_HOST_PTR  
} ocl_mem_flags;

typedef enum {
	MEM_OBJECT_BUFFER  = CL_MEM_OBJECT_BUFFER,
	MEM_OBJECT_IMAGE2D = CL_MEM_OBJECT_IMAGE2D,
	MEM_OBJECT_IMAGE3D = CL_MEM_OBJECT_IMAGE3D
} ocl_mem_object;

typedef enum {
	ERR_FILE_NOT_FOUND = 1000,
	ERR_COULD_NOT_OPEN_FILE
} ocl_custom_errors;

typedef enum {
	BUILD_SUCCESS		= CL_BUILD_SUCCESS,
	BUILD_NONE			= CL_BUILD_NONE,
	BUILD_ERROR			= CL_BUILD_ERROR,
	BUILD_IN_PROGRESS	= CL_BUILD_IN_PROGRESS
} ocl_build_status;

}}
#endif
