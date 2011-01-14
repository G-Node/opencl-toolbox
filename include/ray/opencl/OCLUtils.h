#ifndef _RAY_OPENCL_OCLUTILS_H_
#define _RAY_OPENCL_OCLUTILS_H_

/*
 * OpenCL Utility functions 
 *
 * Author: Radford Juang 
 * Date:  5.7.2010
 * Email: rayver /_at_/ hkn (dot) berkeley (dot) edu
 */

#include <CL/cl.h>
#include <ray/opencl/OCLError.h>

/*
 *  These are helper macros for fetching property information
 *    id	   is the id of the object
 *    propname is the property name to retrieve
 *    dstvar   is the destination variable
 *    dsttype  is the destination type
 *    f        is the associated function to retrieve the information
 */
#define ocl_get_info(id, propname, dstvar, dsttype, f)  { \
	ocl_check( f(id, propname, sizeof(dsttype), &dstvar, NULL), #f " - " #propname ); \
}

#define ocl_get_info_string(id, propname, dstvar, dsttype, f)  {\
	size_t _nbytes;\
	ocl_check( f(id, propname, 0, NULL, &_nbytes), #f " - " #propname ); \
	std::vector<char> _buffer; \
	_buffer.resize(_nbytes / sizeof(char)); \
	ocl_check( f(id, propname, _nbytes, &_buffer[0], NULL), #f " - " #propname ); \
	dstvar = (char *) &_buffer[0];\
}

#define ocl_get_info_vector(id, propname, dstvar, dsttype, f)  { \
	size_t _nbytes;\
	ocl_check( f(id, propname, 0, NULL, &_nbytes), #f " - " #propname ); \
	dstvar.resize(_nbytes/sizeof(dsttype)); \
	ocl_check( f(id, propname, _nbytes, &dstvar[0], NULL), #f " - " #propname ); \
}

/*
 * Helper class for release and retain
 *
 */

namespace ray { namespace opencl {

template <typename T>
class OCLObject {
	protected:
		T  m_id;

	public:
		OCLObject()      : m_id(0)  { }
		OCLObject(T id)  : m_id(id) { retain(); }
		~OCLObject()			    { if (m_id) release(); }
		inline virtual T id()		{ return m_id; }
		inline virtual T *ptr()		{ return &m_id; }

		//To assign an id to an object without retaining it
		// (the class becomes responsible for releasing the object
		inline void	assign(T id ) {  if (m_id) release(); m_id = id; }

	protected:		
		inline void release();
		inline void retain();
		
		inline OCLObject<T>& operator = (const OCLObject<T>& rhs)
		{			
			if (m_id != NULL) { release(); }
			m_id  = rhs.m_id ;

			if (m_id != NULL) { retain(); }

			return *this;
		}
		

};

template<> inline void OCLObject<cl_device_id>::release()   { m_id = 0; }
template<> inline void OCLObject<cl_device_id>::retain()    {			 }

template<> inline void OCLObject<cl_platform_id>::release() { m_id = 0; }
template<> inline void OCLObject<cl_platform_id>::retain()  {			 }

template<> inline void OCLObject<cl_context>::retain()   { ocl_check(clRetainContext(m_id), "clRetainContext"); }
template<> inline void OCLObject<cl_context>::release()  { ocl_check(clReleaseContext(m_id), "clReleaseContext"); m_id = 0; }

template<> inline void OCLObject<cl_command_queue>::retain()   { ocl_check(clRetainCommandQueue(m_id), "clRetainCommandQueue"); }
template<> inline void OCLObject<cl_command_queue>::release()  { ocl_check(clReleaseCommandQueue(m_id), "clReleaseCommandQueue"); m_id = 0; }

template<> inline void OCLObject<cl_mem>::retain()   { ocl_check(clRetainMemObject(m_id),"clRetainMemObject"); }
template<> inline void OCLObject<cl_mem>::release()  { ocl_check(clReleaseMemObject(m_id),"clReleaseMemObject"); m_id = 0; }

template<> inline void OCLObject<cl_program>::retain()   { ocl_check(clRetainProgram(m_id),"clRetainProgram"); }
template<> inline void OCLObject<cl_program>::release()  { ocl_check(clReleaseProgram(m_id),"clReleaseProgram"); m_id = 0; }

template<> inline void OCLObject<cl_kernel>::retain()   { ocl_check(clRetainKernel(m_id),"clRetainKernel"); }
template<> inline void OCLObject<cl_kernel>::release()  { ocl_check(clReleaseKernel(m_id),"clReleaseKernel"); m_id = 0; }

template<> inline void OCLObject<cl_event>::retain()   { ocl_check(clRetainEvent(m_id),"clRetainEvent"); }
template<> inline void OCLObject<cl_event>::release()  { ocl_check(clReleaseEvent(m_id),"clReleaseEvent"); m_id = 0; }

template<> inline void OCLObject<cl_sampler>::retain()   { ocl_check(clRetainSampler(m_id),"clRetainSampler"); }
template<> inline void OCLObject<cl_sampler>::release()  { ocl_check(clReleaseSampler(m_id),"clReleaseSampler"); m_id = 0; }

}}
#endif
