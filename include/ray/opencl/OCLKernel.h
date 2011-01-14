#ifndef _RAY_OPENCL_OCLKERNEL_H_
#define _RAY_OPENCL_OCLKERNEL_H_

/*
 * OpenCL Kernel object for holding the source/binary code program
 * 
 * A kernel object is responsible for holding a snippet of the program 
 * (a function from a Program object) and also maintaining the arguments.
 *
 *
 * Author: Radford Juang 
 * Date:  4.27.2010
 * Email: rayver /_at_/ hkn (dot) berkeley (dot) edu 
 */

#include <CL/cl.h>
#include <ray/opencl/OCLConstants.h>
#include <ray/opencl/OCLError.h>
#include <ray/opencl/OCLProgram.h>
#include <ray/opencl/OCLContext.h>
#include <ray/opencl/OCLUtils.h>
#include <ray/opencl/OCLBuffer.h>

#include <string>
#include <fstream>

namespace ray{ namespace opencl {

typedef struct _OCLKernel_WorkgroupInfo {
	size_t		 work_group_size;
	size_t		 compile_work_group_size[3];
	cl_ulong	 local_mem_size;
} OCLKernel_WorkgroupInfo;

class OCLKernelSizeArg { 
	cl_kernel m_kernel;
	cl_uint	  m_index;	
	size_t	  m_arg_size;
public:
	OCLKernelSizeArg(cl_kernel id, cl_uint idx, size_t arg_size) : 
		m_kernel(id), m_index(idx), m_arg_size(arg_size) {}

	//Set the argument	
	inline void *operator = (void *arg)
	{
		ocl_check_fast(
			clSetKernelArg(m_kernel, m_index, m_arg_size, arg),
			"clSetKernelArg"
		);
		return arg;
	}
};

class OCLKernelArg { 
	cl_kernel m_kernel;
	cl_uint	  m_index;

public:
	OCLKernelArg(cl_kernel id, cl_uint idx) : 
		m_kernel(id), m_index(idx){ }

	//Set the argument	
	template <typename T>
	inline T *operator = (T *arg)
	{
		ocl_check_fast(
			clSetKernelArg(m_kernel, m_index, sizeof(T), arg),
			"clSetKernelArg"
		);
		return arg;
	}

	inline OCLBuffer &operator = (OCLBuffer &arg)
	{ 
		ocl_check_fast(
			clSetKernelArg(m_kernel, m_index, sizeof(cl_mem), reinterpret_cast<void *>(arg.ptr())),
			"clSetKernelArg"
		);
		return arg;
	}

	inline OCLBuffer *operator = (OCLBuffer *arg)
	{
		ocl_check_fast(
			clSetKernelArg(m_kernel, m_index, sizeof(cl_mem), reinterpret_cast<void *>(arg->ptr())),
			"clSetKernelArg"
		);
		return arg;
	}
};

class OCLKernel : public OCLObject<cl_kernel> {
public:
	std::string  m_function_name;
	cl_uint		 m_num_args;
	cl_uint		 m_refcount;
	cl_context   m_context;
	cl_program	 m_program;

	
	cl_uint		 m_num_dims;
	size_t		 m_global_group_offset[3];
	size_t		 m_global_group_size[3];
	size_t		 m_local_group_size[3];

public:
	OCLKernel() : OCLObject<cl_kernel>() { }

	OCLKernel(cl_kernel id) : OCLObject<cl_kernel>(id) { query_info(); }

	OCLKernel(cl_program prog, const char *name) :
		m_program(prog), m_function_name(name) 
	{
		cl_int errcode  = CL_SUCCESS;
		m_id = clCreateKernel(m_program, name, &errcode);
		ocl_check( errcode, "clCreateKernel" );
	}
	

	OCLKernel(OCLProgram &prog, const char *name) :
		m_program(prog.id()), m_function_name(name) 
	{
		cl_int errcode  = CL_SUCCESS;
		m_id = clCreateKernel(m_program, name, &errcode);
		ocl_check( errcode, "clCreateKernel" );
	}

	OCLKernel(OCLProgram *prog, const char *name) :
		m_program(prog->id()), m_function_name(name) 
	{
		cl_int errcode  = CL_SUCCESS;
		m_id = clCreateKernel(m_program, name, &errcode);
		ocl_check( errcode, "clCreateKernel" );
	}

	inline void query_info() {
		ocl_get_info(m_id, CL_KERNEL_NUM_ARGS,		  m_num_args, cl_uint,		clGetKernelInfo);
		ocl_get_info(m_id, CL_KERNEL_REFERENCE_COUNT, m_refcount, cl_uint,		clGetKernelInfo);
		ocl_get_info(m_id, CL_KERNEL_CONTEXT,		  m_context,  cl_context,	clGetKernelInfo);
		ocl_get_info(m_id, CL_KERNEL_PROGRAM,		  m_program,  cl_program,	clGetKernelInfo);
		ocl_get_info_string(m_id, CL_KERNEL_FUNCTION_NAME,	 m_function_name,  std::string,	clGetKernelInfo);
	}

	inline OCLKernel_WorkgroupInfo get_workgroup_info(cl_device_id device) {
		OCLKernel_WorkgroupInfo w;
		cl_int errcode = CL_SUCCESS;
		w.compile_work_group_size[0] = 0;
		w.compile_work_group_size[1] = 0;
		w.compile_work_group_size[2] = 0;

		size_t retsize = 0;
		ocl_check(
			clGetKernelWorkGroupInfo(m_id, device, CL_KERNEL_WORK_GROUP_SIZE,		  sizeof(size_t), &w.work_group_size, NULL),
			"clGetKernelWorkGroupInfo -- CL_KERNEL_WORK_GROUP_SIZE"
		);
		ocl_check(
			clGetKernelWorkGroupInfo(m_id, device, CL_KERNEL_COMPILE_WORK_GROUP_SIZE, 0, 0, &retsize),
			"clGetKernelWorkGroupInfo -- CL_KERNEL_COMPILE_WORK_GROUP_SIZE size"
		);
		ocl_check(
			clGetKernelWorkGroupInfo(m_id, device, CL_KERNEL_COMPILE_WORK_GROUP_SIZE, retsize, w.compile_work_group_size, NULL),
			"clGetKernelWorkGroupInfo -- CL_KERNEL_COMPILE_WORK_GROUP_SIZE"
		);
		ocl_check(
			clGetKernelWorkGroupInfo(m_id, device, CL_KERNEL_LOCAL_MEM_SIZE, sizeof(cl_ulong), &w.local_mem_size,		  NULL),
			"clGetKernelWorkGroupInfo -- CL_KERNEL_LOCAL_MEM_SIZE"
		);

		return w;
	}

	inline void set(cl_uint idx, size_t byte_size, void *value=NULL){
		ocl_check_fast(
			clSetKernelArg(m_id, idx, byte_size, value),
			"clSetKernelArg");
	}

	inline void set_ndims(cl_uint n) {  
		m_num_dims = n;	
	}

	inline void set_global_size(size_t x1, size_t x2=0, size_t x3=0) { 	
		m_global_group_size[0]=x1;
		m_global_group_size[1]=x2;
		m_global_group_size[2]=x3;	
	}

	inline void set_global_offset(size_t x1, size_t x2=0, size_t x3=0 ) {
		m_global_group_offset[0]=x1;
		m_global_group_offset[1]=x2;
		m_global_group_offset[2]=x3;	
	}
	inline void set_local_size(size_t x1, size_t x2=0, size_t x3=0 ) { 
		m_local_group_size[0]=x1;
		m_local_group_size[1]=x2;
		m_local_group_size[2]=x3;	
	}

	inline OCLKernelArg operator() (cl_uint idx) {		
		return OCLKernelArg(m_id, idx);
	}

	inline OCLKernelSizeArg operator() (cl_uint idx, cl_uint size) {		
		return OCLKernelSizeArg(m_id, idx, size);
	}

	inline OCLKernelArg operator[] (cl_uint idx) {		
		return OCLKernelArg(m_id, idx);
	}

};

}}

#endif
