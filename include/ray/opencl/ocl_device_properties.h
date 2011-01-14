#ifndef _RAY_OPENCL_OCL_DEVICE_PROPERTIES_H_
#define _RAY_OPENCL_OCL_DEVICE_PROPERTIES_H_

/*
 * OpenCL Device object properties 
 * Author: Radford Juang 
 * Date:  4.28.2010
 * Email: rayver /_at_/ hkn (dot) berkeley (dot) edu
 */

// Device information available:
#include <CL/cl.h>
#include <string>

namespace ray { namespace opencl {

typedef struct _ocl_device_properties { 
		cl_device_type						type;
		cl_uint								vendor_id;
		cl_uint								max_compute_units;
		cl_uint								max_work_item_dimensions;
		size_t								max_work_group_size;
		std::vector<size_t> 				max_work_item_sizes;
		cl_uint								preferred_vector_width_char;
		cl_uint								preferred_vector_width_short;
		cl_uint								preferred_vector_width_int;
		cl_uint								preferred_vector_width_long;
		cl_uint								preferred_vector_width_float;
		cl_uint								preferred_vector_width_double;
		cl_uint								max_clock_frequency;	
		cl_platform_id						platform;
		cl_bitfield		    				address_bits;
		cl_uint			    				max_read_image_args;
		cl_uint								max_write_image_args;
		cl_long								max_mem_alloc_size;
		size_t								image2d_max_width;
		size_t								image2d_max_height;
		size_t								image3d_max_width;
		size_t								image3d_max_height;
		size_t								image3d_max_depth;
		cl_uint								image_support;
		size_t								max_parameter_size;
		cl_uint								max_samplers;
		cl_uint								mem_base_addr_align;
		cl_uint								min_data_type_align_size;	
		cl_device_fp_config					single_fp_config;
		cl_device_mem_cache_type			global_mem_cache_type;
		cl_uint								global_mem_cacheline_size;
		cl_ulong							global_mem_cache_size;
		cl_ulong							global_mem_size;
		cl_ulong							max_constant_buffer_size;
		cl_uint								max_constant_args;
		cl_device_local_mem_type			local_mem_type;
		cl_ulong							local_mem_size;
		cl_bool								error_correction_support;
		size_t								profiling_timer_resolution;
		cl_bool								endian_little;
		cl_bool								available;
		cl_bool								compiler_available;
		cl_device_exec_capabilities			execution_capabilities;
		cl_command_queue_properties			queue_properties;			

		std::string							name;
		std::string							vendor;
		std::string							version;
		std::string							driver_version;
		std::string							profile;
		std::string							extensions;	
} ocl_device_properties;

}}

#endif
