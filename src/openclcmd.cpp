/*
 * Copyright (C) 2011 by Radford Ray Juang 
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files 
 * (the "Software"), to deal in the Software without restriction, including 
 * without limitation the rights to use, copy, modify, merge, publish, 
 * distribute, sublicense, and/or sell copies of the Software, and to 
 * permit persons to whom the Software is furnished to do so, subject to 
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
 * DEALINGS IN THE SOFTWARE.
 *
 */

/*
 * Author: Radford Ray Juang
 * Email: rayver /_at_/ hkn (dot) berkeley (dot) edu
 */
#include <ray/opencl/opencl.h>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <string.h>

#include "mex.h"

using namespace ray::opencl;

//#define dbg_printf  printf    //Enable debug print statements
#define dbg_printf              //Remove debug print statements

/****************************************************
 * GLOBAL VARIABLES DESIGNATING STATE OF MEX FILE   *
 ****************************************************/

static OCLPlatform *g_platform = 0;            //Pointer to platform to use.
static OCLContext  *g_context  = 0;            //Pointer to context to use.
static OCLProgram  *g_program  = 0;            //Pointer to program (kernels) to load and compile to device


static std::vector<OCLBuffer *> g_buffers;           //Vector of buffers
static std::vector<OCLCommandQueue*> g_queues;       //Vector of pointers to command queues
static std::vector<OCLKernel*> g_kernels;            //Vector of pointers to kernels
static std::vector<unsigned int> g_free_buffer_pool; //Array of free indices in g_buffers


/********************************
 * CLEANUP FUNCTION             *
 ********************************/
static void cleanup(void) {
    //Do cleanup here
    dbg_printf("Closing device...\n");
    delete g_program;
    
    for (int i=0; i<g_queues.size(); ++i) {
        delete g_queues[i];
        g_queues[i] = 0;
    }

    for (int i=0; i<g_kernels.size(); ++i) {
        delete g_kernels[i];
        g_kernels[i] = 0;        
    }

    for (int i=0; i<g_buffers.size(); ++i) {
        delete g_buffers[i];
        g_buffers[i] = 0;        
    }

    g_kernels.clear();
    g_queues.clear();
    g_buffers.clear();
    g_free_buffer_pool.clear();

    delete g_context;
    delete g_platform;

    g_context = 0;
    g_platform = 0;
    g_program = 0;
}

/********************************
 * FUNCTION PROTOTYPES          *
 ********************************/
void fetch_opencl_devices(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
void initialize(mxArray *plhs[], const mxArray *platform, const mxArray *devices);
void add_file(mxArray *plhs[], const mxArray *filename);
void build(mxArray *plhs[]);

void create_buffer(mxArray *plhs[], const mxArray *mode, const mxArray *sz);
void set_buffer(mxArray *plhs[], const mxArray *deviceNumber, const mxArray *bufferNumber, const mxArray *data);
void get_buffer(mxArray *plhs[], const mxArray *deviceNumber, const mxArray *bufferNumber, 
    const mxArray *num_elements, const mxArray *type);
void wait_queue(mxArray *plhs[], const mxArray *deviceNumber);
void create_kernels(mxArray *plhs[], const mxArray *local, const mxArray *global, const mxArray *name);
void execute_kernel(mxArray *plhs[], const mxArray *device_id, const mxArray *kernel_id);

void set_kernel_args(mxArray *plhs[], const mxArray *kernel_id, 
    const mxArray *arg_num, const mxArray *buffer_id, const mxArray *data, const mxArray *size);

void destroy_buffer(mxArray *plhs[], const mxArray *bufferId);

/********************************
 * MAIN MEX FUNCTION            *
 ********************************/
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

    mexAtExit(cleanup);

    if (nrhs < 1) {
        //openclcmd() :
        // -  If no commands passed in, return device information
        fetch_opencl_devices(nlhs, plhs, nrhs, prhs);
        return;
    }

    int nChars = mxGetN(prhs[0])+1;  //Add one extra for the \0
    std::vector<char> buffer;
    buffer.resize(nChars);
    
    mxGetString(prhs[0], &buffer[0], nChars);
    
    buffer[nChars-1] = 0;
    if (strcmp(&buffer[0], "initialize") == 0) {
        //openclcmd('initialize', platform, devices) 
        //  platform: single integer representing the index of platform to use
        //  (zero-centered)
        //  devices: array of integers representing index of devices to use
        //  (zero-centered)
        //
        //Returns true if success, false otherwise.
        if (nrhs < 3)  
            mexErrMsgIdAndTxt("MATLAB:openclcmd:nInput", "Not enough input arguments");

        initialize(plhs, prhs[1], prhs[2]);

    } else if (strcmp(&buffer[0], "addfile") == 0) {
        //openclcmd('addfile', filenames) 
        //  filenames: a cell array of strings containing the filenames of the
        //  OpenCL files to load
        //
        //Returns true if success, false otherwise.
        if (nrhs < 2)
            mexErrMsgIdAndTxt("MATLAB:openclcmd:nInput", "Not enough input arguments");

        add_file(plhs, prhs[1]);

    } else if (strcmp(&buffer[0], "build") == 0) {
        //openclcmd('build')
        //  Compiles and builds the the program. 
        //
        //  Returns true if success, false otherwise
        build(plhs);
    } else if (strcmp(&buffer[0], "create_buffer") == 0) {
        //openclcmd('create_buffer', mode, size)
        //  Create a buffer of a given mode type and size   
        //      mode: 2-character string that is 'rw', 'ro', 'wo', for
        //          read-write, read-only, and write-only 
        //      size: positive integer specifying size of buffer
        //      
        //  Returns -1 if failed, or a number indicating the ID (index value)
        //  of the buffer
        if (nrhs < 3)
            mexErrMsgIdAndTxt("MATLAB:openclcmd:nInput", "Not enough input arguments");

        create_buffer(plhs, prhs[1], prhs[2]);   
    } else if (strcmp(&buffer[0], "destroy_buffer") == 0) {
        //openclcmd('destroy_buffer', buffer_id)
        //  Destroy the buffer with buffer_id and free up any allocated resources   
        //      buffer_id: integer greater than or equal to 0 specifying the index
	//        of the buffer to delete.
	//
        //  Returns true if success
        if (nrhs < 2)
            mexErrMsgIdAndTxt("MATLAB:openclcmd:nInput", "Not enough input arguments");

        destroy_buffer(plhs, prhs[1]);   

    } else if (strcmp(&buffer[0], "set_buffer") == 0 ) {
        //openclcmd('set_buffer', device_idx, buffer_idx, data)
        //    device_idx:  zero-based index containing index of device in
        //      context to use  (e.g. 0 for first device)
        //    buffer_idx: zero-based index containing index of buffer to use
        //    data: matrix containing data to copy to GPU (must cast to
        //      right type to correctly interpret data storage)
        //
        //Returns true if success, false otherwise.
        if (nrhs < 4)
            mexErrMsgIdAndTxt("MATLAB:openclcmd:nInput", "Not enough input arguments");

        set_buffer(plhs, prhs[1], prhs[2], prhs[3]);

    } else if (strcmp(&buffer[0], "get_buffer") == 0 ) {
        //openclcmd('get_buffer', device_idx, buffer_idx, nElems, type)
        //
        //    device_idx: zero-based index containing index of device in
        //      context to use  (e.g. 0 for first device)
        //    buffer_idx: zero-based index containing index of buffer to use
        //    nElems: number of data elements to extract
        //    type: string containing the data type which can be
        //          'int8',  'uint8',
        //          'int16', 'uint16', 
        //          'int32', 'uint32',
        //          'int64', 'uint64',
        //          'single', 'double'
        //          'char', 'logical'
        //
        //Returns a single row vector containing the data.
        if (nrhs < 5)
            mexErrMsgIdAndTxt("MATLAB:openclcmd:nInput", "Not enough input arguments");

        get_buffer(plhs, prhs[1], prhs[2], prhs[3], prhs[4]);
        
    } else if (strcmp(&buffer[0], "create_kernel") == 0 ) {
        //openclcmd('create_kernel', local_dims, global_dims, kernel_name)
        //
        // Create a kernel given the local dimensions, global dimensions, and 
        // kernel name.
        //
        // local_dims must be a uint32 1x3 matrix containing the number of
        //    threads to divide the task into
        // global_dims must be a uint32 1x3 matrix containing the number of 
        //    units to divide the task into
        //
        // Returns an index number (>= 0) containing the ID of the kernel.
        //  -1 if failed.
        
        if (nrhs < 4)
            mexErrMsgIdAndTxt("MATLAB:openclcmd:nInput", "Not enough input arguments");

        create_kernels(plhs, prhs[1], prhs[2], prhs[3]);

    } else if (strcmp(&buffer[0], "set_kernel_args") == 0) {
        //Setting kernel argument to buffer:
        //  openclcmd('set_kernel_args',  kernel_id, arg_num, buffer_id, [], 0 ) 
        //
        //Setting kernel argument to constant data:
        //  openclcmd('set_kernel_args',  kernel_id, arg_num, -1, data, 0 )
        //
        //Setting kernel argument to local variable type:
        //  openclcmd('set_kernel_args',  kernel_id, arg_num, -1, [], nBytes )
        //

        if (nrhs < 6)
            mexErrMsgIdAndTxt("MATLAB:openclcmd:nInput", "Not enough input arguments");
        set_kernel_args(plhs, prhs[1], prhs[2], prhs[3], prhs[4], prhs[5]);

    } else if (strcmp(&buffer[0], "execute_kernel") == 0 ) {
        //openclcmd('execute_kernel', device_id, kernel_id)
        //
        //Execute a kernel given the device and the kernel 
        //
        //device_id : zero-based index of the device number in the lits of
        //  devices in the context (if only 1 device in context, this is 0)
        //kernel_id : zero-based index of the kernel number returned from
        //  create_kernel call
        //
        // returns true if success, false if failed.
         
        if (nrhs < 3)
            mexErrMsgIdAndTxt("MATLAB:openclcmd:nInput", "Not enough input arguments");

        execute_kernel(plhs, prhs[1], prhs[2]);

    } else if (strcmp(&buffer[0], "wait_queue") == 0) {
        //openclcmd('wait_queue', device_idx)
        //    device_idx = zero-based index containing index of device in
        //      context to use  (e.g. 0 for first device)
        //  
        //Returns true if success, false otherwise.
        if (nrhs < 2)
            mexErrMsgIdAndTxt("MATLAB:openclcmd:nInput", "Not enough input arguments");

        wait_queue(plhs, prhs[1]);

    } else if (strcmp(&buffer[0], "cleanup") == 0) {
        //openclcmd('cleanup'): Perform cleanup
        //
        cleanup();
    } else {
        mexErrMsgIdAndTxt("MATLAB:openclcmd:command", "Invalid command");
    }
}

/********************************
 * HELPER SUBROUTINES           *
 ********************************/
void fetch_opencl_devices(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{    
    
    mwSize dims[2] = {1, 1};
    
    //Our output fieldnames:
    const char *platform_field_names[] = {        
        "profile",
        "name",         
        "vendor",    
        "version",     
        "extensions",      
        "devices", 
    };
    
    const char *device_field_names[] = {        
        "profile",
        "name",         
        "vendor",    
        "version",
        "driver",
        "extensions",
        "max_compute_units",
        "max_work_item_dimensions",
        "max_work_group_size",
        "max_work_item_sizes",
        "preferred_vector_width_char",
        "preferred_vector_width_short",
        "preferred_vector_width_int",
        "preferred_vector_width_long",
        "preferred_vector_width_float",
        "preferred_Vector_width_double",
        "max_clock_frequency",
        "address_bits",
        "max_read_image_args",
        "max_write_image_args",
        "max_mem_alloc_size",
        "image2d_max_width",
        "image2d_max_height",
        "image3d_max_width",
        "image3d_max_height",
        "image3d_max_depth",
        "image_support",
        "max_parameter_size",
        "max_samplers",
        "mem_base_addr_align",
        "min_data_type_align_size",
//        "single_fp_config",
//        "global_mem_cache_type",
        "global_mem_cacheline_size",
        "global_mem_cache_size",
        "global_mem_size",
        "max_constant_buffer_size",
        "max_constant_args",
//        "local_mem_type",
        "local_mem_size",
        "error_correction_support",
        "profiling_timer_resolution",
        "endian_little",
        "available",
        "compiler_available",
        "platform"
    };
    
    enum { 
        PLAT_FIELD_PROFILE = 0,     
        PLAT_FIELD_NAME,
        PLAT_FIELD_VENDOR,
        PLAT_FIELD_VERSION,
        PLAT_FIELD_EXTENSIONS,
        PLAT_FIELD_DEVICES,
        PLAT_NUM_FIELDS
    };
    
    enum { 
        DEV_FIELD_PROFILE = 0,     
        DEV_FIELD_NAME,
        DEV_FIELD_VENDOR,
        DEV_FIELD_VERSION,
        DEV_FIELD_DRIVER,
        DEV_FIELD_EXTENSIONS,
        DEV_FIELD_MAX_COMPUTE_UNITS,
        DEV_FIELD_MAX_WORK_ITEM_DIMENSIONS,
        DEV_FIELD_MAX_WORK_GROUP_SIZE,
        DEV_FIELD_MAX_WORK_ITEM_SIZES,
        DEV_FIELD_PREFERRED_VECTOR_WIDTH_CHAR,
        DEV_FIELD_PREFERRED_VECTOR_WIDTH_SHORT,
        DEV_FIELD_PREFERRED_VECTOR_WIDTH_INT,
        DEV_FIELD_PREFERRED_VECTOR_WIDTH_LONG,
        DEV_FIELD_PREFERRED_VECTOR_WIDTH_FLOAT,
        DEV_FIELD_PREFERRED_VECTOR_WIDTH_DOUBLE,
        DEV_FIELD_MAX_CLOCK_FREQUENCY,
        DEV_FIELD_ADDRESS_BITS,
        DEV_FIELD_MAX_READ_IMAGE_ARGS,
        DEV_FIELD_MAX_WRITE_IMAGE_ARGS,
        DEV_FIELD_MAX_MEM_ALLOC_SIZE,
        DEV_FIELD_IMAGE2D_MAX_WIDTH,
        DEV_FIELD_IMAGE2D_MAX_HEIGHT,
        DEV_FIELD_IMAGE3D_MAX_WIDTH,
        DEV_FIELD_IMAGE3D_MAX_HEIGHT,
        DEV_FIELD_IMAGE3D_MAX_DEPTH,
        DEV_FIELD_IMAGE_SUPPORT,
        DEV_FIELD_MAX_PARAMETER_SIZE,
        DEV_FIELD_MAX_SAMPLERS,
        DEV_FIELD_MEM_BASE_ADDR_ALIGN,
        DEV_FIELD_MIN_DATA_TYPE_ALIGN_SIZE,
//        DEV_FIELD_SINGLE_FP_CONFIG,
//        DEV_FIELD_GLOBAL_MEM_CACHE_TYPE,
        DEV_FIELD_GLOBAL_MEM_CACHELINE_SIZE,
        DEV_FIELD_GLOBAL_MEM_CACHE_SIZE,
        DEV_FIELD_GLOBAL_MEM_SIZE,
        DEV_FIELD_MAX_CONSTANT_BUFFER_SIZE,
        DEV_FIELD_MAX_CONSTANT_ARGS,
//        DEV_FIELD_LOCAL_MEM_TYPE,
        DEV_FIELD_LOCAL_MEM_SIZE,
        DEV_FIELD_ERROR_CORRECTION_SUPPORT,
        DEV_FIELD_PROFILING_TIMER_RESOLUTION,
        DEV_FIELD_ENDIAN_LITTLE,
        DEV_FIELD_AVAILABLE,
        DEV_FIELD_COMPILER_AVAILABLE,
        DEV_NUM_FIELDS
    };
    
    dims[0] = 1; dims[1] = 1;
    mxArray *arr;

    try {	
		std::vector<cl_platform_id> platforms = OCLPlatform::get_platform_ids();
        mwSize dims[2] = {platforms.size(), 1};
        
        plhs[0] = mxCreateStructArray(2, dims, PLAT_NUM_FIELDS, platform_field_names);        
        
		for (size_t i=0; i<platforms.size(); ++i) {
			OCLPlatform p(platforms[i]);
			cl_platform_id pid = p.id();
            
            arr = mxCreateString(p.m_profile.c_str()); 
                mxSetField(plhs[0], i, platform_field_names[PLAT_FIELD_PROFILE],  arr);
                
            arr = mxCreateString(p.m_name.c_str()); 
                mxSetField(plhs[0], i, platform_field_names[PLAT_FIELD_NAME],  arr);  
                        
            arr = mxCreateString(p.m_vendor.c_str()); 
                mxSetField(plhs[0], i, platform_field_names[PLAT_FIELD_VENDOR],  arr);  
                        
            arr = mxCreateString(p.m_version.c_str()); 
                mxSetField(plhs[0], i, platform_field_names[PLAT_FIELD_VERSION],  arr);  
                                   
            arr = mxCreateString(p.m_extensions.c_str()); 
                mxSetField(plhs[0], i, platform_field_names[PLAT_FIELD_EXTENSIONS],  arr);              
                        

             //Now fetch devices
            std::vector<cl_device_id> devices = p.get_device_ids();
            
                dims[0] = devices.size();  dims[1] = 1;
                mxArray *dev_arr = mxCreateStructArray(2, dims, DEV_NUM_FIELDS, device_field_names);
            			                        
			for (size_t j=0; j<devices.size(); ++j) {
				OCLDevice d(devices[j]);
                                    
                arr = mxCreateString(d.m_properties.profile.c_str()); 
                    mxSetField(dev_arr, j, device_field_names[DEV_FIELD_PROFILE],  arr);
                
                arr = mxCreateString(d.m_properties.name.c_str()); 
                    mxSetField(dev_arr, j, device_field_names[DEV_FIELD_NAME],  arr);  
                        
                arr = mxCreateString(d.m_properties.vendor.c_str()); 
                    mxSetField(dev_arr, j, device_field_names[DEV_FIELD_VENDOR],  arr);  
                        
                arr = mxCreateString(d.m_properties.version.c_str()); 
                    mxSetField(dev_arr, j, device_field_names[DEV_FIELD_VERSION],  arr);  
                
                arr = mxCreateString(d.m_properties.driver_version.c_str()); 
                    mxSetField(dev_arr, j, device_field_names[DEV_FIELD_DRIVER],  arr);                  
                                   
                arr = mxCreateString(d.m_properties.extensions.c_str());                 
                    mxSetField(dev_arr, j, device_field_names[DEV_FIELD_EXTENSIONS],  arr);

                arr = mxCreateDoubleScalar(d.m_properties.max_compute_units);
                    mxSetField(dev_arr, j, device_field_names[DEV_FIELD_MAX_COMPUTE_UNITS], arr);
              
                arr = mxCreateDoubleScalar(d.m_properties.max_work_item_dimensions);
                    mxSetField(dev_arr, j, device_field_names[DEV_FIELD_MAX_WORK_ITEM_DIMENSIONS], arr);

                arr = mxCreateDoubleScalar(d.m_properties.max_work_group_size);
                    mxSetField(dev_arr, j, device_field_names[DEV_FIELD_MAX_WORK_GROUP_SIZE], arr);

                arr = mxCreateDoubleMatrix(1, d.m_properties.max_work_item_dimensions, mxREAL);
                    double *pdata = (double *) mxGetData(arr);

                    for (int k=0; k<d.m_properties.max_work_item_dimensions; ++k) {
                        pdata[k] = d.m_properties.max_work_item_sizes[k];
                    }
                    mxSetField(dev_arr, j, device_field_names[DEV_FIELD_MAX_WORK_ITEM_SIZES], arr);

                arr = mxCreateDoubleScalar(d.m_properties.preferred_vector_width_char);
                    mxSetField(dev_arr, j, device_field_names[DEV_FIELD_PREFERRED_VECTOR_WIDTH_CHAR], arr);

                arr = mxCreateDoubleScalar(d.m_properties.preferred_vector_width_short);
                    mxSetField(dev_arr, j, device_field_names[DEV_FIELD_PREFERRED_VECTOR_WIDTH_SHORT], arr);

                arr = mxCreateDoubleScalar(d.m_properties.preferred_vector_width_int);
                    mxSetField(dev_arr, j, device_field_names[DEV_FIELD_PREFERRED_VECTOR_WIDTH_INT], arr);

                arr = mxCreateDoubleScalar(d.m_properties.preferred_vector_width_long);
                    mxSetField(dev_arr, j, device_field_names[DEV_FIELD_PREFERRED_VECTOR_WIDTH_LONG], arr);

                arr = mxCreateDoubleScalar(d.m_properties.preferred_vector_width_float);
                    mxSetField(dev_arr, j, device_field_names[DEV_FIELD_PREFERRED_VECTOR_WIDTH_FLOAT], arr);

                arr = mxCreateDoubleScalar(d.m_properties.preferred_vector_width_double);
                    mxSetField(dev_arr, j, device_field_names[DEV_FIELD_PREFERRED_VECTOR_WIDTH_DOUBLE], arr);
    
                arr = mxCreateDoubleScalar(d.m_properties.max_clock_frequency);
                    mxSetField(dev_arr, j, device_field_names[DEV_FIELD_MAX_CLOCK_FREQUENCY], arr);

                arr = mxCreateDoubleScalar(d.m_properties.address_bits);
                    mxSetField(dev_arr, j, device_field_names[DEV_FIELD_ADDRESS_BITS], arr);

                arr = mxCreateDoubleScalar(d.m_properties.max_read_image_args);
                    mxSetField(dev_arr, j, device_field_names[DEV_FIELD_MAX_READ_IMAGE_ARGS], arr);

                arr = mxCreateDoubleScalar(d.m_properties.max_write_image_args);
                    mxSetField(dev_arr, j, device_field_names[DEV_FIELD_MAX_WRITE_IMAGE_ARGS], arr);

                arr = mxCreateDoubleScalar(d.m_properties.max_mem_alloc_size);
                    mxSetField(dev_arr, j, device_field_names[DEV_FIELD_MAX_MEM_ALLOC_SIZE], arr);

                arr = mxCreateDoubleScalar(d.m_properties.image2d_max_width);
                    mxSetField(dev_arr, j, device_field_names[DEV_FIELD_IMAGE2D_MAX_WIDTH], arr);

                arr = mxCreateDoubleScalar(d.m_properties.image2d_max_height);
                    mxSetField(dev_arr, j, device_field_names[DEV_FIELD_IMAGE2D_MAX_HEIGHT], arr);

                arr = mxCreateDoubleScalar(d.m_properties.image3d_max_width);
                    mxSetField(dev_arr, j, device_field_names[DEV_FIELD_IMAGE3D_MAX_WIDTH], arr);

                arr = mxCreateDoubleScalar(d.m_properties.image3d_max_height);
                    mxSetField(dev_arr, j, device_field_names[DEV_FIELD_IMAGE3D_MAX_HEIGHT], arr);

                arr = mxCreateDoubleScalar(d.m_properties.image3d_max_depth);
                    mxSetField(dev_arr, j, device_field_names[DEV_FIELD_IMAGE3D_MAX_DEPTH], arr);

                arr = mxCreateDoubleScalar(d.m_properties.image_support);
                    mxSetField(dev_arr, j, device_field_names[DEV_FIELD_IMAGE_SUPPORT], arr);

                arr = mxCreateDoubleScalar(d.m_properties.max_parameter_size);
                    mxSetField(dev_arr, j, device_field_names[DEV_FIELD_MAX_PARAMETER_SIZE], arr);

                arr = mxCreateDoubleScalar(d.m_properties.max_samplers);
                    mxSetField(dev_arr, j, device_field_names[DEV_FIELD_MAX_SAMPLERS], arr);

                arr = mxCreateDoubleScalar(d.m_properties.mem_base_addr_align);
                    mxSetField(dev_arr, j, device_field_names[DEV_FIELD_MEM_BASE_ADDR_ALIGN], arr);

                arr = mxCreateDoubleScalar(d.m_properties.min_data_type_align_size);
                    mxSetField(dev_arr, j, device_field_names[DEV_FIELD_MIN_DATA_TYPE_ALIGN_SIZE], arr);

                //arr = mxCreateDoubleScalar(d.m_properties.single_fp_config);
                //    mxSetField(dev_arr, j, device_field_names[DEV_FIELD_SINGLE_FP_CONFIG], arr);

                //arr = mxCreateDoubleScalar(d.m_properties.global_mem_cache_type);
                //    mxSetField(dev_arr, j, device_field_names[DEV_FIELD_GLOBAL_MEM_CACHE_TYPE], arr);

                arr = mxCreateDoubleScalar(d.m_properties.global_mem_cacheline_size);
                    mxSetField(dev_arr, j, device_field_names[DEV_FIELD_GLOBAL_MEM_CACHELINE_SIZE], arr);

                arr = mxCreateDoubleScalar(d.m_properties.global_mem_cache_size);
                    mxSetField(dev_arr, j, device_field_names[DEV_FIELD_GLOBAL_MEM_CACHE_SIZE], arr);
   
                arr = mxCreateDoubleScalar(d.m_properties.global_mem_size);
                    mxSetField(dev_arr, j, device_field_names[DEV_FIELD_GLOBAL_MEM_SIZE], arr);

                arr = mxCreateDoubleScalar(d.m_properties.max_constant_buffer_size);
                    mxSetField(dev_arr, j, device_field_names[DEV_FIELD_MAX_CONSTANT_BUFFER_SIZE], arr);

                arr = mxCreateDoubleScalar(d.m_properties.max_constant_args);
                    mxSetField(dev_arr, j, device_field_names[DEV_FIELD_MAX_CONSTANT_ARGS], arr);

                //arr = mxCreateDoubleScalar(d.m_properties.local_mem_type);
                //    mxSetField(dev_arr, j, device_field_names[DEV_FIELD_LOCAL_MEM_TYPE], arr);

                arr = mxCreateDoubleScalar(d.m_properties.local_mem_size);
                    mxSetField(dev_arr, j, device_field_names[DEV_FIELD_LOCAL_MEM_SIZE], arr);

                arr = mxCreateDoubleScalar(d.m_properties.error_correction_support);
                    mxSetField(dev_arr, j, device_field_names[DEV_FIELD_ERROR_CORRECTION_SUPPORT], arr);

                arr = mxCreateDoubleScalar(d.m_properties.profiling_timer_resolution);
                    mxSetField(dev_arr, j, device_field_names[DEV_FIELD_PROFILING_TIMER_RESOLUTION], arr);

                arr = mxCreateDoubleScalar(d.m_properties.endian_little);
                    mxSetField(dev_arr, j, device_field_names[DEV_FIELD_ENDIAN_LITTLE], arr);

                arr = mxCreateDoubleScalar(d.m_properties.available);
                    mxSetField(dev_arr, j, device_field_names[DEV_FIELD_AVAILABLE], arr);

                arr = mxCreateDoubleScalar(d.m_properties.compiler_available);
                    mxSetField(dev_arr, j, device_field_names[DEV_FIELD_COMPILER_AVAILABLE], arr);
    
			}
            mxSetField(plhs[0], i, platform_field_names[PLAT_FIELD_DEVICES],  dev_arr);                                  
		}
        
	} catch (OCLError err) {        
		std::cout << "fetch_opencl_devices: Error " << err.m_code << ": " << err.m_message << " (" << err.m_notes << ")" << std::endl;
        mexErrMsgTxt("Runtime error! (See error message above)");        
	} catch (...) {
		std::cout << "fetch_opencl_devices: Unknown error occurred!" << std::endl;
        mexErrMsgTxt("Runtime error! (See error message above)");        
	}        
}

void initialize(mxArray *plhs[], const mxArray *platform, const mxArray *devices) {
    int platform_idx = static_cast<int>(mxGetScalar(platform) );
    int len = 0;
    unsigned int *p_data_uint32 = 0;

    int return_value = 0;
    int device_idx = 0;
    try {
    	std::vector<cl_platform_id> platforms = OCLPlatform::get_platform_ids();
        
        //Establish platform and context
        if (g_platform != 0) cleanup();
       
        dbg_printf("# Platforms: %d\n", platforms.size());
        dbg_printf("Connecting to platform %d: ", platform_idx);
        g_platform = new OCLPlatform(platforms[platform_idx]); 
        dbg_printf("OK\n");

        dbg_printf("Opening context: ");
        g_context = new OCLContext(*g_platform);
        dbg_printf("OK\n");

        dbg_printf("Adding devices: \n");
        std::vector<cl_device_id> available_devices = g_platform->get_device_ids(); 

        len = mxGetNumberOfElements(devices);
        p_data_uint32 = reinterpret_cast<unsigned int *>( mxGetData(devices));  

    	for (size_t j=0; j<len; ++j) { 
            device_idx = p_data_uint32[j];
            *g_context += available_devices[device_idx];
    	}

        g_context->create();

        dbg_printf("Creating program object: \n");
        g_program = new OCLProgram(*g_context);

        g_queues.resize(len);
        for (size_t j=0; j<len; ++j) {
            device_idx = p_data_uint32[j];
            g_queues[j] = new OCLCommandQueue(*g_context, available_devices[device_idx]);
        }

        return_value = 1;
        dbg_printf("OK\n");
    } catch(OCLError err) {
        dbg_printf("FAIL\n");
		std::cout << "initialize: Error " << err.m_code << ": " << err.m_message << " (" << err.m_notes << ")" << std::endl;
        mexErrMsgTxt("Runtime error! (See error message above)");        
    } catch (...) {
        dbg_printf("FAIL\n");
		std::cout << "initialize: Unknown error occurred!" << std::endl;
        mexErrMsgTxt("Runtime error! (See error message above)");        
    }

    plhs[0] = mxCreateLogicalScalar(return_value);    
}

void add_file(mxArray *plhs[], const mxArray *filename) {
    int return_value = 0;
    try {
        dbg_printf("Add sources or binaries: \n");
    	//Add sources or binaries
        std::vector<char> kernel_filename;
        int len = mxGetNumberOfElements(filename);   
        kernel_filename.resize(len+1);
        mxGetString(filename, &kernel_filename[0], len+1);        

    	g_program->add_source(&kernel_filename[0]);
        return_value = 1;
    } catch(OCLError err) {
        dbg_printf("FAIL\n");
        std::cout << "add_file: Error " << err.m_code << ": " << err.m_message << " (" << err.m_notes << ")" << std::endl;
        mexErrMsgTxt("Runtime error! (See error message above)");        
    } catch (...) {
        dbg_printf("FAIL\n");
        std::cout << "add_file: Unknown error occurred!" << std::endl;
        mexErrMsgTxt("Runtime error! (See error message above)");        
    }
    plhs[0] = mxCreateLogicalScalar(return_value);
}

void build(mxArray *plhs[]) {
    int return_value = 0;
    try {
        g_program->create();
        g_program->build();

        //Check to see if there are any errors:
        std::string err_msg;
        int berror = 0;

        for (int i=0; i < g_program->m_build_status.size(); ++i) {
            if (g_program->m_build_status[i].status == CL_BUILD_ERROR) {
                //Collect all error messages:
                berror = 1;
                err_msg += "Error: \n";
                err_msg += g_program->m_build_status[i].log;
                err_msg += "\n";
            }
        }

        if (berror) {
            throw OCLError(CL_BUILD_ERROR, err_msg.c_str());
        }

        return_value = 1;
    } catch(OCLError err) {
        dbg_printf("FAIL\n");
        std::cout << "build: Error " << err.m_code << ": " << err.m_message << " (" << err.m_notes << ")" << std::endl;
        mexErrMsgTxt("Runtime error! (See error message above)");        
    } catch (...) {
        dbg_printf("FAIL\n");
        std::cout << "build: Unknown error occurred!" << std::endl;
        mexErrMsgTxt("Runtime error! (See error message above)");        
    }

    plhs[0] = mxCreateLogicalScalar(return_value);
}

void create_buffer(mxArray *plhs[], const mxArray *mode, const mxArray *sz) {
    int len;

    len = mxGetNumberOfElements(mode);
    std::vector<char> buf;
    buf.resize(len+1);
    mxGetString(mode, &buf[0], len+1);
    int flags;
    int nSz;

    nSz = static_cast<int>(mxGetScalar(sz));

    if (buf[0] == 'r') {
        if (buf[1] == 'o') {
            flags = MEM_FLAGS_READ_ONLY;
        } else if (buf[1] == 'w') {
            flags = MEM_FLAGS_READ_WRITE;
        } else {

        }
    } else if (buf[0] == 'w') {
        if (buf[1] == 'o') {
            flags = MEM_FLAGS_WRITE_ONLY;
        } else {

        }
    } 

    len = -1;
    try {        
        len = g_buffers.size();
        dbg_printf("Size of buffer = %d\n", len);
       
        OCLBuffer *b = new OCLBuffer(*g_context, flags, nSz);

	    //Check to see if we have a free buffer index first:
	    if (g_free_buffer_pool.empty()) {
             g_buffers.push_back(b);
	    } else {
	        unsigned int freeidx = g_free_buffer_pool[g_free_buffer_pool.size()-1];
	        g_free_buffer_pool.pop_back();
	        g_buffers[freeidx] = b;
	        len = static_cast<int>(freeidx);
        } 
    } catch (OCLError err) {
        dbg_printf("FAIL\n");
        std::cout << "create_buffer: Error " << err.m_code << ": " << err.m_message << " (" << err.m_notes << ")" << std::endl;
        mexErrMsgTxt("Runtime error! (See error message above)");        
    } catch (...) {
        dbg_printf("FAIL\n");
        std::cout << "create_buffer: Unknown error occurred!" << std::endl;
        mexErrMsgTxt("Runtime error! (See error message above)");        
    }
    plhs[0] = mxCreateDoubleScalar(len);
}

void destroy_buffer(mxArray *plhs[], const mxArray *buffer_id) {
    unsigned int idx = static_cast<int>(mxGetScalar(buffer_id) );

    int returnval = 0;
    try {
	    if ((idx < 0) || (idx >= g_buffers.size())) {
	        //throw idx;
	        return;  //Already de-allocated
	    }

	    if (g_buffers[idx] == 0) return;  //Already de-allocated. 
	
	    delete g_buffers[idx];
	    g_buffers[idx] = 0;
	    g_free_buffer_pool.push_back(idx);
	    returnval = 1;
    } catch (OCLError err) {
        dbg_printf("FAIL\n");
        std::cout << "destroy_buffer: Error " << err.m_code << ": " << err.m_message << " (" << err.m_notes << ")" << std::endl;
        mexErrMsgTxt("Runtime error! (See error message above)");        
    } catch (unsigned int i) {
	    std::cout << "Invalid buffer id value " << i << std::endl;
	    std::cout << "Current buffer size " << g_buffers.size() << std::endl;
        mexErrMsgTxt("Runtime error! (See error message above)");        
    } catch (...) {
        dbg_printf("FAIL\n");
        std::cout << "destroy_buffer: Unknown error occurred!" << std::endl;
        mexErrMsgTxt("Runtime error! (See error message above)");        
    }
    plhs[0] = mxCreateLogicalScalar(returnval);
}

void set_buffer(mxArray *plhs[], const mxArray *deviceNumber, const mxArray *bufferNumber, const mxArray *data) {
    int sz = mxGetNumberOfElements(data);
    if (mxIsDouble(data)) {
        sz *= 8;
    } else if (mxIsSingle(data)) {
        sz *= 4;
    } else if (mxIsInt8(data) || mxIsUint8(data)) {
        sz *= 1;
    } else if (mxIsInt16(data) || mxIsUint16(data)) {
        sz *= 2;
    } else if (mxIsInt32(data) || mxIsUint32(data)) {
        sz *= 4;
    } else if (mxIsInt64(data) || mxIsUint64(data)) {
        sz *= 8;
    }
    void *pData = mxGetData(data);
    size_t buf_idx = (size_t) mxGetScalar(bufferNumber);
    size_t dev_idx = (size_t) mxGetScalar(deviceNumber);

    int return_val = 0;
    try {
        g_queues[dev_idx]->enqueue_buffer_copy(*g_buffers[buf_idx], pData, sz);
        g_queues[dev_idx]->finish();
        return_val = 1;
    } catch(OCLError err) {
        dbg_printf("FAIL\n");
        std::cout << "set_buffer: Error " << err.m_code << ": " << err.m_message << " (" << err.m_notes << ")" << std::endl;
        mexErrMsgTxt("Runtime error! (See error message above)");        
    } catch(...) {
        dbg_printf("FAIL\n");
        std::cout << "set_buffer: Unknown error occurred!" << std::endl;
        mexErrMsgTxt("Runtime error! (See error message above)");        
    }
    plhs[0] = mxCreateLogicalScalar(return_val);
}

void wait_queue(mxArray *plhs[], const mxArray *deviceNumber) {
    size_t dev_idx = (size_t) mxGetScalar(deviceNumber);
    int return_val = 0;

    try{
        g_queues[dev_idx]->finish();
        return_val = 1;
    } catch(OCLError err) {
        dbg_printf("FAIL\n");
        std::cout << "wait_queue: Error " << err.m_code << ": " << err.m_message << " (" << err.m_notes << ")" << std::endl;
        mexErrMsgTxt("Runtime error! (See error message above)");        
    } catch(...) {
        dbg_printf("FAIL\n");
        std::cout << "wait_queue: Unknown error occurred!" << std::endl;
        mexErrMsgTxt("Runtime error! (See error message above)");        
    }

    plhs[0] = mxCreateLogicalScalar(return_val);
}

void get_buffer(mxArray *plhs[], const mxArray *deviceNumber, const mxArray *bufferNumber, 
    const mxArray *num_elements, const mxArray *type) {
    size_t dev_idx = (size_t) mxGetScalar(deviceNumber);
    size_t buf_idx = (size_t) mxGetScalar(bufferNumber);
   
    size_t sz = (size_t) mxGetScalar(num_elements);

    int len = mxGetNumberOfElements(type);

    std::vector<char> type_str;
    type_str.resize(len+1);
    mxGetString(type, &type_str[0], len+1);

    mxArray *arr = 0;

    size_t mrows = 1;
    size_t nElems = sz;

    if (strcmp(&type_str[0], "int8") == 0) {                
         arr = mxCreateNumericMatrix(mrows,nElems, mxINT8_CLASS, mxREAL);
         sz *= 1;
    } else if (strcmp(&type_str[0], "int16") == 0) {
         sz *= 2;
         arr = mxCreateNumericMatrix(mrows,nElems, mxINT16_CLASS, mxREAL);                                
    } else if (strcmp(&type_str[0], "int32") == 0) {
         sz *= 4;
         arr = mxCreateNumericMatrix(mrows,nElems, mxINT32_CLASS, mxREAL);
    } else if (strcmp(&type_str[0], "int64") == 0) {
         sz *= 8;
         arr = mxCreateNumericMatrix(mrows,nElems, mxINT64_CLASS, mxREAL);
    } else if (strcmp(&type_str[0], "uint8") == 0) {
         sz *= 1;
         arr = mxCreateNumericMatrix(mrows,nElems, mxUINT8_CLASS, mxREAL);
    } else if (strcmp(&type_str[0], "uint16") == 0) {
         sz *= 2;
         arr = mxCreateNumericMatrix(mrows,nElems, mxUINT16_CLASS, mxREAL);
    } else if (strcmp(&type_str[0], "uint32") == 0) {
         sz *= 4;
         arr = mxCreateNumericMatrix(mrows,nElems, mxUINT32_CLASS, mxREAL);                
    } else if (strcmp(&type_str[0], "uint64") == 0) {
         sz *= 8;
         arr = mxCreateNumericMatrix(mrows,nElems, mxUINT64_CLASS, mxREAL);
    } else if (strcmp(&type_str[0], "single") == 0) {
         sz *= 4;
         arr = mxCreateNumericMatrix(mrows,nElems, mxSINGLE_CLASS, mxREAL);
    } else if (strcmp(&type_str[0], "double") == 0) {
         sz *= 8;
         arr = mxCreateNumericMatrix(mrows,nElems, mxDOUBLE_CLASS, mxREAL);                
    } else if (strcmp(&type_str[0], "char") == 0) {
         mwSize dims[2] = {mrows, nElems};                
         arr = mxCreateCharArray(2, dims);
    } else if (strcmp(&type_str[0], "logical") == 0) {
         mwSize dims[2] = {mrows, nElems};                
         arr = mxCreateLogicalArray(2, dims);                
    } else {
        dbg_printf("FAIL\n");
        std::cout << "get_buffer: Unsupported data type!" << std::endl;
        mexErrMsgTxt("Runtime error! (See error message above)");        
        //Unsupported!
        return;
    }

    try {
        void *dst = mxGetData(arr); 
        g_queues[dev_idx]->enqueue_buffer_copy(dst, *g_buffers[buf_idx], sz, 0, CL_FALSE);
        g_queues[dev_idx]->finish(); //When a copy occurs, need to wait before returning.. otherwise, crash will happen
        plhs[0] = arr;
    } catch(OCLError err) {
        dbg_printf("FAIL\n");
        std::cout << "get_buffer: Error " << err.m_code << ": " << err.m_message << " (" << err.m_notes << ")" << std::endl;
        mexErrMsgTxt("Runtime error! (See error message above)");        
    } catch(...) {
        dbg_printf("FAIL\n");
        std::cout << "get_buffer: Unknown error occurred!" << std::endl;
        mexErrMsgTxt("Runtime error! (See error message above)");        
    }
}

void create_kernels(mxArray *plhs[], const mxArray *local, const mxArray *global, const mxArray *name) {
    //Require local and global to be cast to uint32!

    unsigned int global_size[3] = {0,0,0};
    unsigned int local_size[3] = {0,0,0};

    unsigned int *pdata;
    int len;

    pdata = (unsigned int *) mxGetData(local);
    len = mxGetNumberOfElements(local);
    if (len > 3) len = 3;
    for (int i=0; i<len; ++i) local_size[i] = pdata[i];

    pdata = (unsigned int *) mxGetData(global);
    len = mxGetNumberOfElements(global);
    if (len > 3) len = 3;
    for (int i=0; i<len; ++i) global_size[i] = pdata[i];

    std::vector<char> kernel_name;
    len = mxGetNumberOfElements(name);
    kernel_name.resize(len+1);
    mxGetString(name, &kernel_name[0], len+1);

    int ndims = 0;
    for (ndims =0; (ndims < 3) && (global_size[ndims] > 0) ; ++ndims) {}

    try {
        OCLKernel *kernel = new OCLKernel(*g_program, &kernel_name[0]);
		kernel->set_global_offset(0,0,0);
		kernel->set_ndims(ndims);
		kernel->set_local_size(local_size[0],local_size[1],local_size[2]);
        kernel->set_global_size(global_size[0],global_size[1],global_size[2]);
        len = g_kernels.size();
        g_kernels.resize(len+1);
        g_kernels[len] = kernel;
    } catch(OCLError err) {
        dbg_printf("FAIL\n");
        std::cout << "create_kernels: Error " << err.m_code << ": " << err.m_message << " (" << err.m_notes << ")" << std::endl;
        mexErrMsgTxt("Runtime error! (See error message above)");        
        len = -1;
    } catch(...) {
        dbg_printf("FAIL\n");
        std::cout << "create_kernels: Unknown error occurred!" << std::endl;
        mexErrMsgTxt("Runtime error! (See error message above)");        
        len = -1;
    }

    plhs[0] = mxCreateDoubleScalar(len);
}

void execute_kernel(mxArray *plhs[], const mxArray *device_id, const mxArray *kernel_id) {
    size_t dev_idx = (size_t) mxGetScalar(device_id);
    size_t kernel_idx = (size_t) mxGetScalar(kernel_id);
   
    int return_val = 0;
    try {
        g_queues[dev_idx]->enqueue_ndrange_kernel(g_kernels[kernel_idx]);
        return_val = 1;
    } catch(OCLError err) {
        dbg_printf("FAIL\n");
        std::cout << "execute_kernel: Error " << err.m_code << ": " << err.m_message << " (" << err.m_notes << ")" << std::endl;
        mexErrMsgTxt("Runtime error! (See error message above)");        
    } catch(...) {
        dbg_printf("FAIL\n");
        std::cout << "execute_kernel: Unknown error occurred!" << std::endl;
        mexErrMsgTxt("Runtime error! (See error message above)");        
    }
    plhs[0] = mxCreateLogicalScalar(return_val);
}

//set_kernel_args( kernel_id, arg_num, buffer_id, [], 0 )    arg: buffer
//set_kernel_args( kernel_id, arg_num, -1, data, 0 )         arg: constant data
//set_kernel_args( kernel_id, arg_num, -1, [], nBytes )      arg: local variable
//
void set_kernel_args(mxArray *plhs[], const mxArray *kernel_id, 
    const mxArray *arg_num, const mxArray *buffer_id, const mxArray *data, const mxArray *size) {

    size_t kernel_idx = (size_t) mxGetScalar(kernel_id);
    size_t arg_idx = (size_t) mxGetScalar(arg_num);
    int buf_idx = (int) mxGetScalar(buffer_id);
    void *pdata = 0;
    int sz = 0;


    if (!mxIsEmpty(data)) {
        //Local type. Compute size
        pdata = mxGetData(data);

        sz = mxGetNumberOfElements(data);
        if (mxIsDouble(data)) {
            sz *= 8;
        } else if (mxIsSingle(data)) {
            sz *= 4;
        } else if (mxIsInt8(data) || mxIsUint8(data)) {
            sz *= 1;
        } else if (mxIsInt16(data) || mxIsUint16(data)) {
            sz *= 2;
        } else if (mxIsInt32(data) || mxIsUint32(data)) {
            sz *= 4;
        } else if (mxIsInt64(data) || mxIsUint64(data)) {
            sz *= 8;
        }

    } else {
        //Local type. Needs an explicit size:
        sz = (int) mxGetScalar(size);
    }

    int return_val = 0;
    try {
        if (buf_idx >= 0) {
            (*(g_kernels[kernel_idx]))[arg_idx] = *g_buffers[buf_idx];
        } else {
            (*(g_kernels[kernel_idx]))(arg_idx, sz ) = pdata;
        }
        return_val = 1;
    } catch(OCLError err) {
        dbg_printf("FAIL\n");
        std::cout << "set_kernel_args: Error " << err.m_code << ": " << err.m_message << " (" << err.m_notes << ")" << std::endl;
        mexErrMsgTxt("Runtime error! (See error message above)");        
    } catch(...) {
        dbg_printf("FAIL\n");
        std::cout << "set_kernel_args: Unknown error occurred!" << std::endl;
        mexErrMsgTxt("Runtime error! (See error message above)");        
    }
    plhs[0] = mxCreateLogicalScalar(return_val);
}

