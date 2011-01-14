#ifndef _RAY_OPENCL_OCLPLATFORM_H_
#define _RAY_OPENCL_OCLPLATFORM_H_

/*
 * OpenCL Platform object for decoding information from a 
 * provided platform_id. 
 * 
 * A platform on OpenCL is the software platform interface 
 * binding software and hardware. 
 *
 * For example, if you have an AMD device and 2 NVIDIA devices,
 * there will be two different platforms. One platform for AMD, 
 * and one platform for NVIDIA. 
 * 
 *
 * Author: Radford Juang 
 * Date:  5.6.2010
 * Email: rayver /_at_/ hkn (dot) berkeley (dot) edu
 */

#include <CL/cl.h>
#include <vector>
#include <string>

#include <ray/opencl/OCLConstants.h>
#include <ray/opencl/OCLError.h>
#include <ray/opencl/OCLUtils.h>

namespace ray { namespace opencl {

class OCLPlatform : public OCLObject<cl_platform_id> {
public:
	std::string m_profile;
	std::string m_version;
	std::string m_name;
	std::string m_vendor;	
	std::string m_extensions;	

public:
		
	OCLPlatform(cl_platform_id id) : OCLObject<cl_platform_id>(id) 
	{			
		ocl_get_info_string(m_id, CL_PLATFORM_PROFILE,	m_profile,	  std::string, clGetPlatformInfo);
		ocl_get_info_string(m_id, CL_PLATFORM_VERSION,	m_version,	  std::string, clGetPlatformInfo);
		ocl_get_info_string(m_id, CL_PLATFORM_NAME,		m_name,		  std::string, clGetPlatformInfo);
		ocl_get_info_string(m_id, CL_PLATFORM_VENDOR,	m_vendor,		  std::string, clGetPlatformInfo);
		ocl_get_info_string(m_id, CL_PLATFORM_EXTENSIONS, m_extensions, std::string, clGetPlatformInfo);		
	}

	inline std::vector<cl_device_id> get_device_ids(ocl_device_type type = DEVICE_TYPE_ALL ) {
		std::vector<cl_device_id> devices;
		cl_uint num_devices;
		
		ocl_check( 
			clGetDeviceIDs(m_id, type, 0, NULL, &num_devices),
			"clGetDeviceIDs" 
		);

		devices.resize(num_devices);		

		ocl_check(
			clGetDeviceIDs(m_id, type, num_devices, &devices[0], 0),
			"clGetDeviceIDs"
		);

		return devices;
	}

public:
	inline static std::vector<cl_platform_id> get_platform_ids() {		
		std::vector<cl_platform_id> platform_list;
		
		cl_uint num_platforms = 0;
		ocl_check(
			clGetPlatformIDs(0, NULL, &num_platforms),
			"clGetPlatformIDs");

		platform_list.resize(num_platforms);
		ocl_check(
			clGetPlatformIDs(num_platforms, &platform_list[0], NULL),
			"clGetPlatformIDs");

		return platform_list;
	}
};

}}


#endif
