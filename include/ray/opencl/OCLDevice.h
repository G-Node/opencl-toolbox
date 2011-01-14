#ifndef _RAY_OPENCL_OCLDEVICE_H_
#define _RAY_OPENCL_OCLDEVICE_H_

/*
 * OpenCL Device object for querying OpenCL device information from 
 *  a cl_device_id
 *
 * Author: Radford Juang 
 * Date:  5.6.2010
 * Email: rayver /_at_/ hkn (dot) berkeley (dot) edu
 */

#include <CL/cl.h>

#include <ray/opencl/ocl_device_properties.h>
#include <ray/opencl/OCLError.h>
#include <ray/opencl/OCLUtils.h>

namespace ray { namespace opencl {

class OCLDevice : public OCLObject<cl_device_id>{
public:						
	ocl_device_properties				m_properties;	// Various properties of the device

public:
	OCLDevice(cl_device_id id) : OCLObject<cl_device_id>(id) {
	//Search & replace query:
	//	F\(cl_device_info,[\t ]*CL_{[:alpha_]+},[\t ]*{[:alpha_\:<> ]+}[\t ]*\)[\t ]*\\
	//	ocl_get_info(m_id,CL_\(-37,1), m_properties.\(-37,1), \(-20,2), clGetDeviceInfo);
		ocl_get_info(m_id,CL_DEVICE_TYPE                          , m_properties.type                          , cl_device_type      , clGetDeviceInfo);
		ocl_get_info(m_id,CL_DEVICE_VENDOR_ID                     , m_properties.vendor_id                     , cl_uint             , clGetDeviceInfo);
		ocl_get_info(m_id,CL_DEVICE_MAX_COMPUTE_UNITS             , m_properties.max_compute_units             , cl_uint             , clGetDeviceInfo);
		ocl_get_info(m_id,CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS      , m_properties.max_work_item_dimensions      , cl_uint             , clGetDeviceInfo);
		ocl_get_info(m_id,CL_DEVICE_MAX_WORK_GROUP_SIZE           , m_properties.max_work_group_size           , size_t              , clGetDeviceInfo);
		ocl_get_info_vector(m_id,CL_DEVICE_MAX_WORK_ITEM_SIZES    , m_properties.max_work_item_sizes           , size_t		  		, clGetDeviceInfo);
		ocl_get_info(m_id,CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR   , m_properties.preferred_vector_width_char   , cl_uint             , clGetDeviceInfo);
		ocl_get_info(m_id,CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT  , m_properties.preferred_vector_width_short  , cl_uint             , clGetDeviceInfo);
		ocl_get_info(m_id,CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT    , m_properties.preferred_vector_width_int    , cl_uint             , clGetDeviceInfo);
		ocl_get_info(m_id,CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG   , m_properties.preferred_vector_width_long   , cl_uint             , clGetDeviceInfo);
		ocl_get_info(m_id,CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT  , m_properties.preferred_vector_width_float  , cl_uint             , clGetDeviceInfo);
		ocl_get_info(m_id,CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE , m_properties.preferred_vector_width_double , cl_uint             , clGetDeviceInfo);
		ocl_get_info(m_id,CL_DEVICE_MAX_CLOCK_FREQUENCY           , m_properties.max_clock_frequency           , cl_uint             , clGetDeviceInfo);
		ocl_get_info(m_id,CL_DEVICE_ADDRESS_BITS                  , m_properties.address_bits                  , cl_bitfield         , clGetDeviceInfo);
		ocl_get_info(m_id,CL_DEVICE_MAX_READ_IMAGE_ARGS           , m_properties.max_read_image_args           , cl_uint             , clGetDeviceInfo);
		ocl_get_info(m_id,CL_DEVICE_MAX_WRITE_IMAGE_ARGS          , m_properties.max_write_image_args          , cl_uint             , clGetDeviceInfo);
		ocl_get_info(m_id,CL_DEVICE_MAX_MEM_ALLOC_SIZE            , m_properties.max_mem_alloc_size            , cl_ulong            , clGetDeviceInfo);
		ocl_get_info(m_id,CL_DEVICE_IMAGE2D_MAX_WIDTH             , m_properties.image2d_max_width             , size_t              , clGetDeviceInfo);
		ocl_get_info(m_id,CL_DEVICE_IMAGE2D_MAX_HEIGHT            , m_properties.image2d_max_height            , size_t              , clGetDeviceInfo);
		ocl_get_info(m_id,CL_DEVICE_IMAGE3D_MAX_WIDTH             , m_properties.image3d_max_width             , size_t              , clGetDeviceInfo);
		ocl_get_info(m_id,CL_DEVICE_IMAGE3D_MAX_HEIGHT            , m_properties.image3d_max_height            , size_t              , clGetDeviceInfo);
		ocl_get_info(m_id,CL_DEVICE_IMAGE3D_MAX_DEPTH             , m_properties.image3d_max_depth             , size_t              , clGetDeviceInfo);
		ocl_get_info(m_id,CL_DEVICE_IMAGE_SUPPORT                 , m_properties.image_support                 , cl_uint             , clGetDeviceInfo);
		ocl_get_info(m_id,CL_DEVICE_MAX_PARAMETER_SIZE            , m_properties.max_parameter_size            , size_t              , clGetDeviceInfo);
		ocl_get_info(m_id,CL_DEVICE_MAX_SAMPLERS                  , m_properties.max_samplers                  , cl_uint             , clGetDeviceInfo);
		ocl_get_info(m_id,CL_DEVICE_MEM_BASE_ADDR_ALIGN           , m_properties.mem_base_addr_align           , cl_uint             , clGetDeviceInfo);
		ocl_get_info(m_id,CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE      , m_properties.min_data_type_align_size      , cl_uint             , clGetDeviceInfo);
		ocl_get_info(m_id,CL_DEVICE_SINGLE_FP_CONFIG              , m_properties.single_fp_config              , cl_device_fp_config , clGetDeviceInfo);
		ocl_get_info(m_id,CL_DEVICE_GLOBAL_MEM_CACHE_TYPE         , m_properties.global_mem_cache_type         , cl_device_mem_cache_type, clGetDeviceInfo);
		ocl_get_info(m_id,CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE     , m_properties.global_mem_cacheline_size     , cl_uint             , clGetDeviceInfo);
		ocl_get_info(m_id,CL_DEVICE_GLOBAL_MEM_CACHE_SIZE         , m_properties.global_mem_cache_size         , cl_ulong            , clGetDeviceInfo);
		ocl_get_info(m_id,CL_DEVICE_GLOBAL_MEM_SIZE               , m_properties.global_mem_size               , cl_ulong            , clGetDeviceInfo);
		ocl_get_info(m_id,CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE      , m_properties.max_constant_buffer_size      , cl_ulong            , clGetDeviceInfo);
		ocl_get_info(m_id,CL_DEVICE_MAX_CONSTANT_ARGS             , m_properties.max_constant_args             , cl_uint             , clGetDeviceInfo);
		ocl_get_info(m_id,CL_DEVICE_LOCAL_MEM_TYPE                , m_properties.local_mem_type                , cl_device_local_mem_type, clGetDeviceInfo);
		ocl_get_info(m_id,CL_DEVICE_LOCAL_MEM_SIZE                , m_properties.local_mem_size                , cl_ulong            , clGetDeviceInfo);
		ocl_get_info(m_id,CL_DEVICE_ERROR_CORRECTION_SUPPORT      , m_properties.error_correction_support      , cl_bool             , clGetDeviceInfo);
		ocl_get_info(m_id,CL_DEVICE_PROFILING_TIMER_RESOLUTION    , m_properties.profiling_timer_resolution    , size_t			     , clGetDeviceInfo);
		ocl_get_info(m_id,CL_DEVICE_ENDIAN_LITTLE                 , m_properties.endian_little                 , cl_bool             , clGetDeviceInfo);
		ocl_get_info(m_id,CL_DEVICE_AVAILABLE                     , m_properties.available                     , cl_bool             , clGetDeviceInfo);
		ocl_get_info(m_id,CL_DEVICE_COMPILER_AVAILABLE            , m_properties.compiler_available            , cl_bool             , clGetDeviceInfo);
		ocl_get_info(m_id,CL_DEVICE_EXECUTION_CAPABILITIES        , m_properties.execution_capabilities        , cl_device_exec_capabilities, clGetDeviceInfo);
		ocl_get_info(m_id,CL_DEVICE_QUEUE_PROPERTIES              , m_properties.queue_properties              , cl_command_queue_properties, clGetDeviceInfo);
		ocl_get_info(m_id,CL_DEVICE_PLATFORM                      , m_properties.platform                      , cl_platform_id      , clGetDeviceInfo);

		ocl_get_info_string(m_id,CL_DEVICE_NAME                          , m_properties.name                   , STRING_CLASS        , clGetDeviceInfo);
		ocl_get_info_string(m_id,CL_DEVICE_VENDOR                        , m_properties.vendor                 , STRING_CLASS        , clGetDeviceInfo);
		ocl_get_info_string(m_id,CL_DRIVER_VERSION                       , m_properties.driver_version         , STRING_CLASS        , clGetDeviceInfo);
		ocl_get_info_string(m_id,CL_DEVICE_PROFILE                       , m_properties.profile                , STRING_CLASS        , clGetDeviceInfo);
		ocl_get_info_string(m_id,CL_DEVICE_VERSION                       , m_properties.version                , STRING_CLASS        , clGetDeviceInfo);
		ocl_get_info_string(m_id,CL_DEVICE_EXTENSIONS                    , m_properties.extensions             , STRING_CLASS        , clGetDeviceInfo);
	}

};

}}

#endif
