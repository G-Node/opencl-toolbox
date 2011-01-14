#ifndef _RAY_OPENCL_OCLCONTEXT_H_
#define _RAY_OPENCL_OCLCONTEXT_H_

/*
 * OpenCL Context object for managing devices
 * 
 * A context on OpenCL is responsible for managing memory and 
 * maintaining communication with the devices. A context
 * is dependent on the platform (think of it like a file handler
 * to the devices in question).
 *  
 * Author: Radford Juang 
 * Date:  5.7.2010
 * E-mail: rayver /_at_/ hkn (dot) berkeley (dot) edu
 */

#include <vector>

#include <ray/opencl/OCLDevice.h>
#include <ray/opencl/OCLUtils.h>
#include <ray/opencl/OCLConstants.h>
#include <ray/opencl/OCLError.h>
#include <ray/opencl/OCLPlatform.h>


namespace ray { namespace opencl {

class OCLContext : public OCLObject<cl_context>{
public:
	std::vector<cl_device_id>		   m_devices;
	std::vector<cl_context_properties> m_properties;
	cl_uint							   m_refcount;


public:

	OCLContext(cl_platform_id platform) {
		add_property(CL_CONTEXT_PLATFORM, platform);
	}

	OCLContext(OCLPlatform &platform) {
		add_property(CL_CONTEXT_PLATFORM, platform.id());
	}


	OCLContext(OCLPlatform *platform) {
		add_property(CL_CONTEXT_PLATFORM, platform->id());
	}

	OCLContext(cl_context id) : OCLObject<cl_context>(id) { query_info(); }

	OCLContext(cl_platform_id platform, ocl_device_type type) {
		cl_context_properties props[] = { CL_CONTEXT_PLATFORM,  (cl_context_properties) platform, 0 };
		int errcode = CL_SUCCESS;
		m_id = clCreateContextFromType(props, type, NULL, NULL, &errcode);
		ocl_check(errcode, "clCreateContextFromType");

		query_info();
	}

	OCLContext(OCLPlatform &platform, ocl_device_type type) {
		cl_context_properties props[] = { CL_CONTEXT_PLATFORM,  (cl_context_properties) platform.id(), 0 };
		int errcode = CL_SUCCESS;
		m_id = clCreateContextFromType(props, type, NULL, NULL, &errcode);
		ocl_check(errcode, "clCreateContextFromType");

		query_info();
	}

	OCLContext(OCLPlatform *platform, ocl_device_type type) {
		cl_context_properties props[] = { CL_CONTEXT_PLATFORM,  (cl_context_properties) platform->id(), 0 };
		int errcode = CL_SUCCESS;
		m_id = clCreateContextFromType(props, type, NULL, NULL, &errcode);
		ocl_check(errcode, "clCreateContextFromType");

		query_info();
	}

	inline cl_context id() { 
		if (!m_id) create();		
		return m_id; 
	}

	template<typename T>
	inline void add_property(cl_context_properties name, T value) {
		m_properties.push_back(name);
		m_properties.push_back(reinterpret_cast<cl_context_properties>(value));
	}

	inline void add_device(cl_device_id device) {
		//First make sure device isn't in our queue:		
		int exists = 0;
		for (size_t j=0; j<m_devices.size(); ++j) {
			if (m_devices[j] == device )  {
				exists = 1;
				break;
			}				
		}
		if (!exists) m_devices.push_back(device);
	}

	inline void add_device(OCLDevice &device) {
		add_device(device.id());
	}

	inline void add_device(OCLDevice *device) {
		add_device(device->id());
	}

	inline void add_device(std::vector<cl_device_id> &devices) {
		for (size_t i=0; i<devices.size(); ++i) {
			add_device(devices[i]);
		}
	}

	inline void add_device(std::vector<OCLDevice> &devices) {
		for (size_t i=0; i<devices.size(); ++i) {
			add_device(devices[i]);
		}
	}

	//Can add a context of devices
	inline OCLContext& operator += (const OCLContext& rhs)
	{
		int nDevices = rhs.m_devices.size();
		for (int i=0; i < nDevices; ++i) 
			add_device(rhs.m_devices[i]);

		return *this;
	}

	//Add a single device
	inline OCLContext& operator += (const cl_device_id& rhs)
	{
		add_device(rhs);
		return *this;
	}

	//Add a single device
	inline OCLContext& operator += (OCLDevice& rhs)
	{
		add_device(rhs);
		return *this;
	}

	//Add a single device
	inline OCLContext& operator += (OCLDevice *rhs)
	{
		add_device(rhs);
		return *this;
	}

	//Add a list of devices
	inline OCLContext& operator += (const std::vector<cl_device_id>& rhs)
	{
		int nDevices = rhs.size();
		for (int i=0; i < nDevices; ++i) 
			add_device(rhs[i]);

		return *this;
	}

	//Add a list of devices
	inline OCLContext& operator += (std::vector<OCLDevice>& rhs)
	{		
		int nDevices = rhs.size();
		for (int i=0; i < nDevices; ++i) 
			add_device(rhs[i]);

		return *this;
	}

	inline void create() {
		if (m_id) release();

		m_properties.push_back(0);

		cl_int errcode;
		m_id = clCreateContext(&m_properties[0], m_devices.size(), &m_devices[0], NULL, NULL, &errcode);
		ocl_check(errcode, "clCreateContext");		
		
		query_info();
	}

protected:
	inline void query_info( ) {
		if (!m_id) return;

		ocl_get_info_vector(m_id, CL_CONTEXT_DEVICES,    m_devices,	   cl_device_id,		  clGetContextInfo);
		ocl_get_info_vector(m_id, CL_CONTEXT_PROPERTIES, m_properties, cl_context_properties, clGetContextInfo);
		ocl_get_info(m_id, CL_CONTEXT_REFERENCE_COUNT,   m_refcount,   cl_uint,				  clGetContextInfo);

		if ((m_properties.size() > 0) &&
			(m_properties[m_properties.size()-1] == 0)) {
			m_properties.pop_back();
		}		
	}

};

}}

#endif
