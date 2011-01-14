#ifndef _RAY_OPENCL_OCLPROGRAM_H_
#define _RAY_OPENCL_OCLPROGRAM_H_

/*
 * OpenCL Program object for holding the source/binary code program
 * 
 * A program object is responsible for storing or maintaining code on the
 * device. A program can be associated with multiple kernel functions within
 * a program.
 *
 * Author: Radford Juang 
 * Date:  5.7.2010
 * Email: rayver /_at_/ hkn (dot) berkeley (dot) edu
 */

#include <CL/cl.h>

#include <ray/opencl/OCLDevice.h>
#include <ray/opencl/OCLConstants.h>
#include <ray/opencl/OCLContext.h>
#include <ray/opencl/OCLError.h>
#include <ray/opencl/OCLUtils.h>

#include <vector>
#include <string>
#include <fstream>

namespace ray{ namespace opencl {

class OCLProgram_BuildInfo {
public:
	OCLProgram_BuildInfo() { }
	OCLProgram_BuildInfo(cl_build_status s) {
		status_string = parse_status(s);
	}

public:
	cl_build_status   status;
	std::string		  status_string;
	std::string		  options;
	std::string		  log;

public:
	static const char *parse_status(cl_build_status s) {
		switch (s) {
			case CL_BUILD_SUCCESS:			return "BUILD SUCCESS";
			case CL_BUILD_NONE:				return "BUILD NONE";
			case CL_BUILD_ERROR:			return "BUILD ERROR";
			case CL_BUILD_IN_PROGRESS:		return "BUILD IN PROGRESS";		
		}

		return "UNKNOWN BUILD STATUS";
	}

};

class OCLProgram : public OCLObject<cl_program> {
public:
	cl_context								m_context;
	std::vector<cl_device_id>				m_devices;	

	std::vector<std::string>				m_source;
	std::vector<std::vector<char> >			m_binary;

	std::vector<int>						m_status;
	std::vector<OCLProgram_BuildInfo>		m_build_status;
	cl_uint									m_refcount;

public:
	OCLProgram(cl_program prog) : OCLObject<cl_program>(prog) {
		query_info();	
	}

	OCLProgram(cl_context context) : m_context(context), OCLObject<cl_program>() { }
	OCLProgram(OCLContext &context) : m_context(context.id()), OCLObject<cl_program>() { }
	OCLProgram(OCLContext *context) : m_context(context->id()), OCLObject<cl_program>() { }

	inline void create() {
		//Prefer binary files if source and binary-device pairing both exist for some odd-reason
		if (m_id) release();

		if ((m_devices.size() > 0) && 
			(m_binary.size() > 0)) {
			create_from_binary();
		} else if (m_source.size() > 0) {
			create_from_source();
		}
	}

	inline void add_source(const char *filename) {
		cl_int errcode = CL_SUCCESS;
		std::ifstream file(filename, std::ios_base::binary | std::ios_base::in);

		if (!file.is_open()) {
			errcode = ERR_COULD_NOT_OPEN_FILE;			
		}
		
		ocl_check(errcode, "OCLProgram::add_source");
				

		std::string text(std::istreambuf_iterator<char>(file), (std::istreambuf_iterator<char>()));
		add_source(text);
	}

	inline void add_source(const std::string &text) {
		m_source.push_back(text);
	}

	inline void add_binary(OCLDevice *device,  const char *filename) {
		add_binary(device->id(), filename);
	}

	inline void add_binary(OCLDevice &device,  const char *filename) {
		add_binary(device.id(), filename);
	}

	inline void add_binary(cl_device_id device, const char *filename) {
		cl_int errcode = CL_SUCCESS;
		std::ifstream file(filename, std::ios_base::binary | std::ios_base::in );

		if (!file.is_open()) {
			errcode = ERR_COULD_NOT_OPEN_FILE;			
		}
		
		ocl_check(errcode, "OCLProgram::add_binary");
		
		std::string binstr(std::istreambuf_iterator<char>(file), (std::istreambuf_iterator<char>()));

		add_binary(device, binstr);	
	}

	inline void add_binary(OCLDevice &device, const std::string &binstr) {
		add_binary(device.id(), binstr);
	}

	inline void add_binary(OCLDevice *device, const std::string &binstr) {
		add_binary(device->id(), binstr);
	}

	inline void add_binary(cl_device_id device, const std::string &binstr) {
		//Add device and binary string
		//First check to see if device exists
		int found_device = 0;
		size_t found_idx = 0;
		for (size_t i=0; i<m_devices.size(); ++i) {
			if (m_devices[i] == device) {
				found_device=1;
				found_idx = i;
				break;
			}
		}

		if (found_device) {
			m_binary[found_idx].assign(binstr.begin(), binstr.end());
		} else {
			std::vector<char> binvec;
			binvec.assign(binstr.begin(), binstr.end());
			m_binary.push_back(binvec);
			m_devices.push_back(device);
		}
	}

	inline void build(const char *build_options = NULL) {
		//If no arguments provided or only build options provided, build on
		// all devices in context
		OCLContext context(m_context);
		build(context.m_devices, build_options);
	}

	inline void build(OCLDevice &device, const char *build_options=NULL) {
		build(device.id(), build_options);
	}

	inline void build(OCLDevice *device, const char *build_options=NULL) {
		build(device->id(), build_options);
	}

	inline void build(cl_device_id device, const char *build_options=NULL) {
		std::vector<cl_device_id> devices;
		devices.push_back(device);
		build(devices, build_options);
	}

	inline void build(std::vector<cl_device_id> &devices, const char *build_options=NULL) {		
		//Don't throw because an error in building is a result of bad code, etc. Show log for error details
		if (!m_id) create();

		clBuildProgram(m_id, devices.size(), &devices[0], build_options, NULL, NULL);	

		ocl_check(clUnloadCompiler(), "clUnloadCompiler");
		query_info();			//When we build, we want to retrieve the binaries  and the build info
		query_build_info();
	}

	std::vector<cl_kernel> get_kernels() {	
		cl_uint num_kernels=0;
		ocl_check(
			clCreateKernelsInProgram(m_id, 0, NULL, &num_kernels),
			"clCreateKernelsInProgram");

		std::vector<cl_kernel> kernels;
		kernels.resize(num_kernels);

		ocl_check(
			clCreateKernelsInProgram(m_id, num_kernels, &kernels[0], NULL),
			"clCreateKernelsInProgram");

		return kernels;
	}

protected:

	inline void query_info() {		
		std::string source = "";
		std::vector<size_t> binary_sizes;

		ocl_get_info		(m_id, CL_PROGRAM_REFERENCE_COUNT,  m_refcount,   cl_uint,		clGetProgramInfo);
		ocl_get_info		(m_id, CL_PROGRAM_CONTEXT,		    m_context,    cl_context,	clGetProgramInfo);
		ocl_get_info_vector	(m_id, CL_PROGRAM_DEVICES,			m_devices,	  cl_device_id,	clGetProgramInfo);
		ocl_get_info_string (m_id, CL_PROGRAM_SOURCE,			source,		  std::string,	clGetProgramInfo);
		ocl_get_info_vector (m_id, CL_PROGRAM_BINARY_SIZES,		binary_sizes, size_t,		clGetProgramInfo);

		//Update our source vector
		m_source.clear();
		if (source.size() > 0) {
			m_source.resize(1);
			m_source[0] = source;		//Result is a concatenation of all sources
		} 		

		//Retrieve binaries. First allocate enough room for each vector:
		m_binary.resize(binary_sizes.size());
		
		std::vector<char *> binaries;
		binaries.resize(binary_sizes.size());

		for (size_t i=0; i<binary_sizes.size(); ++i) {			
			m_binary[i].resize(binary_sizes[i]);
			if (binary_sizes[i] == 0) continue;
			binaries[i] = &m_binary[i][0];
		}

		ocl_get_info_vector (m_id, CL_PROGRAM_BINARIES,		binaries, char *, clGetProgramInfo);
	}

	inline void query_build_info() {
		std::vector<char> buffer;
		m_build_status.resize(m_devices.size());
		OCLProgram_BuildInfo info;
		size_t num_bytes = 0;
		for (size_t i=0; i<m_devices.size(); ++i) {		
			ocl_check(
				clGetProgramBuildInfo(m_id, m_devices[i], CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &info.status, NULL),
				"clGetProgramBuildInfo"
			);

			info.status_string = OCLProgram_BuildInfo::parse_status(info.status);

			ocl_check(
				clGetProgramBuildInfo(m_id, m_devices[i], CL_PROGRAM_BUILD_OPTIONS, 0, NULL, &num_bytes),
				"clGetProgramBuildInfo"
			);

			buffer.resize(num_bytes/sizeof(char));

			ocl_check(
				clGetProgramBuildInfo(m_id, m_devices[i], CL_PROGRAM_BUILD_OPTIONS, num_bytes, &buffer[0], NULL),
				"clGetProgramBuildInfo"
			);

			info.options = &buffer[0];


			ocl_check(
				clGetProgramBuildInfo(m_id, m_devices[i], CL_PROGRAM_BUILD_LOG, 0, NULL, &num_bytes),
				"clGetProgramBuildInfo"
			);

			buffer.resize(num_bytes/sizeof(char));

			ocl_check(
				clGetProgramBuildInfo(m_id, m_devices[i], CL_PROGRAM_BUILD_LOG, num_bytes, &buffer[0], NULL),
				"clGetProgramBuildInfo"
			);

			info.log = &buffer[0];

			m_build_status[i] = info;
		}
	}

	inline void create_from_source() {		
		std::vector<const char *> strings;
		strings.resize(m_source.size());

		std::vector<size_t> lengths;
		lengths.resize(m_source.size());

		for (size_t i=0; i < m_source.size(); ++i) {
			lengths[i] = m_source[i].size(); 
			strings[i] = m_source[i].c_str();			
		}

		cl_int errcode = CL_SUCCESS;
		m_id = clCreateProgramWithSource(m_context, m_source.size(), &strings[0], &lengths[0], &errcode);

		ocl_check(errcode, "clCreateProgramWithSource");	
	}

	inline void create_from_binary() {				
		std::vector<size_t> lengths;
		lengths.resize(m_binary.size());
		m_status.resize(m_binary.size());

		for (size_t i=0; i < m_binary.size(); ++i) {
			lengths[i] = m_binary[i].size();
		}

		cl_int errcode = CL_SUCCESS;
		m_id = clCreateProgramWithBinary(m_context, m_devices.size(), &m_devices[0], &lengths[0], (const unsigned char **) &m_binary[0], &m_status[0], &errcode);

		ocl_check(errcode, "clCreateProgramWithBinary");	
	}
};

}}

#endif
