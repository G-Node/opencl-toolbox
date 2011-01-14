#ifndef _RAY_OPENCL_OCLBUFFER_H_
#define _RAY_OPENCL_OCLBUFFER_H_

/*
 * OpenCL Buffer object for managing memory
 * 
 * A Buffer in OpenCL is responsible for holding kernel code and
 * any data necessary
 *  
 * Author: Radford Ray Juang 
 * Date:  5.7.2010
 * E-mail: rayver /_at_/ hkn (dot) berkeley (dot) edu
 */

#include <ray/opencl/OCLContext.h>
#include <ray/opencl/OCLConstants.h>
#include <ray/opencl/OCLError.h>
#include <ray/opencl/OCLUtils.h>

namespace ray { namespace opencl {

class OCLBuffer : public OCLObject<cl_mem> { 
public:
	cl_context			m_context;

	cl_mem_object_type	m_type;			//Contains the object type
	cl_mem_flags		m_flags;		//Contains the memory flags set
	size_t				m_size;			//Contains the memory size
	void			   *m_host_ptr;			//Contains the host pointer
	cl_uint				m_map_count;	//Contains the map count
	cl_uint				m_refcount;			//Contains the reference count

public:

	//Create object and create determine buffer size and flags later
	OCLBuffer(cl_context context, cl_mem_flags flags = 0) : 
		m_context(context), m_flags(flags), 
		m_host_ptr(0), m_size(0), m_map_count(0), m_refcount(0),
		m_type(CL_MEM_OBJECT_BUFFER)
	{ }		

	OCLBuffer(OCLContext &context, cl_mem_flags flags = 0) : 
		m_context(context.id()), m_flags(flags), 
		m_host_ptr(0), m_size(0), m_map_count(0), m_refcount(0),
		m_type(CL_MEM_OBJECT_BUFFER)
	{ }		

	OCLBuffer(OCLContext *context, cl_mem_flags flags = 0) : 
		m_context(context->id()), m_flags(flags), 
		m_host_ptr(0), m_size(0), m_map_count(0), m_refcount(0),
		m_type(CL_MEM_OBJECT_BUFFER)
	{ }		

    OCLBuffer() { }

	OCLBuffer(cl_mem id) : OCLObject<cl_mem>(id) { query_info(); }

	OCLBuffer(cl_context context, cl_mem_flags flags, size_t num_bytes, void *host_ptr = 0) :		
		m_context(context),
		m_flags(flags), 
		m_size(num_bytes),
		m_host_ptr(host_ptr)
	{
		m_id = 0;
		create();		
	}

	OCLBuffer(OCLContext &context, cl_mem_flags flags, size_t num_bytes, void *host_ptr = 0) :		
		m_context(context.id()),
		m_flags(flags), 
		m_size(num_bytes),
		m_host_ptr(host_ptr)
	{
		m_id = 0;
		create();		
	}

	OCLBuffer(OCLContext *context, cl_mem_flags flags, size_t num_bytes, void *host_ptr = 0) :		
		m_context(context->id()),
		m_flags(flags), 
		m_size(num_bytes),
		m_host_ptr(host_ptr)
	{
		m_id = 0;
		create();		
	}

	inline void set_size(size_t sz)			  { m_size = sz; }
	inline void set_hostptr(void *ptr)		  { m_host_ptr = ptr; }
	inline void set_flags(cl_mem_flags flags) { m_flags = flags; }

	inline void create() {
		if (m_id) release();

		int errcode = CL_SUCCESS;
		m_id = clCreateBuffer(m_context, m_flags, m_size, m_host_ptr, &errcode);
		ocl_check(errcode, "clCreateBuffer");

		query_info();
	}

protected:
	inline void query_info() {
		ocl_get_info(m_id, CL_MEM_TYPE,				m_type,			cl_mem_object_type, clGetMemObjectInfo);
		ocl_get_info(m_id, CL_MEM_FLAGS,			m_flags,		cl_mem_flags,		clGetMemObjectInfo);
		ocl_get_info(m_id, CL_MEM_HOST_PTR,			m_host_ptr,		void *,				clGetMemObjectInfo);
		ocl_get_info(m_id, CL_MEM_REFERENCE_COUNT,	m_refcount,		cl_uint,			clGetMemObjectInfo);
		ocl_get_info(m_id, CL_MEM_SIZE,				m_size,			size_t,				clGetMemObjectInfo);
		ocl_get_info(m_id, CL_MEM_MAP_COUNT,		m_map_count,	cl_uint,			clGetMemObjectInfo);
		ocl_get_info(m_id, CL_MEM_CONTEXT,			m_context,		cl_context,			clGetMemObjectInfo);
	}
};

}}
#endif
