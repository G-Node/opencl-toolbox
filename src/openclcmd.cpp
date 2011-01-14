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

#define dbg_printf  printf

/****************************************************
 * GLOBAL VARIABLES DESIGNATING STATE OF MEX FILE   *
 ****************************************************/

static OCLPlatform *g_platform = 0;            //Pointer to platform to use.
static OCLContext  *g_context  = 0;            //Pointer to context to use.
static OCLProgram  *g_program  = 0;            //Pointer to program (kernels) to load and compile to device


static std::vector<OCLBuffer> g_buffers;       //Vector of buffers
static std::vector<OCLCommandQueue*> g_queues; //Vector of pointers to command queues
static std::vector<OCLKernel*> g_kernels;      //Vector of pointers to kernels

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

    g_kernels.clear();
    g_queues.clear();
    g_buffers.clear();

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
        //  devices: array of integers representing index of devices to use
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
        //  set_kernel_args( kernel_id, arg_num, -1, data, 0 )
        //
        //Setting kernel argument to local variable type:
        //  set_kernel_args( kernel_id, arg_num, -1, [], nBytes )
        //

        if (nrhs < 6)
            mexErrMsgIdAndTxt("MATLAB:openclcmd:nInput", "Not enough input arguments");
        set_kernel_args(plhs, prhs[1], prhs[2], prhs[3], prhs[4], prhs[5]);

    } else if (strcmp(&buffer[0], "execute_kernel") == 0 ) {
        //openclcmd('create_kernel', device_id, kernel_id)
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
        "extensions"
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
                                   
                arr = mxCreateString(p.m_extensions.c_str());                 
                    mxSetField(dev_arr, j, device_field_names[DEV_FIELD_EXTENSIONS],  arr);                   
			}
            mxSetField(plhs[0], i, platform_field_names[PLAT_FIELD_DEVICES],  dev_arr);                                  
		}
        
	} catch (OCLError err) {        
        mexErrMsgTxt("Runtime error:");        
		std::cout << "Error " << err.m_code << ": " << err.m_message << " (" << err.m_notes << ")" << std::endl;
	} catch (...) {
        mexErrMsgTxt("Runtime error:");
		std::cout << "Unknown error occurred!" << std::endl;
	}        
}

void initialize(mxArray *plhs[], const mxArray *platform, const mxArray *devices) {
    int platform_idx = static_cast<int>(mxGetScalar(platform) - 1);
    int len = 0;
    unsigned int *p_data_uint32 = 0;

    int return_value = 0;
    int device_idx = 0;
    try {
    	std::vector<cl_platform_id> platforms = OCLPlatform::get_platform_ids();
        
        //Establish platform and context
        if (g_platform != 0) cleanup();
        
        dbg_printf("Connecting platform: ");
        g_platform = new OCLPlatform(platforms[platform_idx]); 
        dbg_printf("OK\n");

        dbg_printf("Opening context: ");
        g_context = new OCLContext(*g_platform);
        dbg_printf("OK\n");

        dbg_printf("Adding devices: ");
        std::vector<cl_device_id> available_devices = g_platform->get_device_ids(); 

        len = mxGetNumberOfElements(devices);
        p_data_uint32 = reinterpret_cast<unsigned int *>( mxGetData(devices));  

    	for (size_t j=0; j<len; ++j) { 
            device_idx = p_data_uint32[j];
            *g_context += available_devices[device_idx];
    	}

        g_context->create();

        dbg_printf("Creating program object: ");
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
        mexErrMsgTxt("Runtime error:");
		std::cout << "Error " << err.m_code << ": " << err.m_message << " (" << err.m_notes << ")" << std::endl;
    } catch (...) {
        dbg_printf("FAIL\n");
        mexErrMsgTxt("Runtime error:");
		std::cout << "Unknown error occurred!" << std::endl;
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
        mexErrMsgTxt("Runtime error:");
        std::cout << "Error " << err.m_code << ": " << err.m_message << " (" << err.m_notes << ")" << std::endl;
    } catch (...) {
        dbg_printf("FAIL\n");
        mexErrMsgTxt("Runtime error:");
        std::cout << "Unknown error occurred!" << std::endl;
    }
    plhs[0] = mxCreateLogicalScalar(return_value);
}

void build(mxArray *plhs[]) {
    int return_value = 0;
    try {
        g_program->create();
        g_program->build();
        return_value = 1;
    } catch(OCLError err) {
        dbg_printf("FAIL\n");
        mexErrMsgTxt("Runtime error:");
        std::cout << "Error " << err.m_code << ": " << err.m_message << " (" << err.m_notes << ")" << std::endl;
    } catch (...) {
        dbg_printf("FAIL\n");
        mexErrMsgTxt("Runtime error:");
        std::cout << "Unknown error occurred!" << std::endl;
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
        g_buffers.resize(len+1);
    
        OCLBuffer b(*g_context, flags, nSz);
        b.create();
        g_buffers[len] = b;
    } catch (OCLError err) {
        dbg_printf("FAIL\n");
        mexErrMsgTxt("Runtime error:");
        std::cout << "Error " << err.m_code << ": " << err.m_message << " (" << err.m_notes << ")" << std::endl;
    } catch (...) {
        dbg_printf("FAIL\n");
        mexErrMsgTxt("Runtime error:");
        std::cout << "Unknown error occurred!" << std::endl;
    }
    plhs[0] = mxCreateDoubleScalar(len);
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
        g_queues[dev_idx]->enqueue_buffer_copy(g_buffers[buf_idx], pData, sz);
        return_val = 1;
    } catch(OCLError err) {
        dbg_printf("FAIL\n");
        mexErrMsgTxt("Runtime error:");
        std::cout << "Error " << err.m_code << ": " << err.m_message << " (" << err.m_notes << ")" << std::endl;
    } catch(...) {
        dbg_printf("FAIL\n");
        mexErrMsgTxt("Runtime error:");
        std::cout << "Unknown error occurred!" << std::endl;
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
        mexErrMsgTxt("Runtime error:");
        std::cout << "Error " << err.m_code << ": " << err.m_message << " (" << err.m_notes << ")" << std::endl;
    } catch(...) {
        dbg_printf("FAIL\n");
        mexErrMsgTxt("Runtime error:");
        std::cout << "Unknown error occurred!" << std::endl;
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
    size_t nElems = 1;

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
        mexErrMsgTxt("Runtime error:");
        std::cout << "Unsupported data type!" << std::endl;
        //Unsupported!
        return;
    }

    try {
        void *dst = mxGetData(arr); 
        g_queues[dev_idx]->enqueue_buffer_copy(dst, g_buffers[buf_idx], sz, 0, CL_FALSE);
        plhs[0] = arr;
    } catch(OCLError err) {
        dbg_printf("FAIL\n");
        mexErrMsgTxt("Runtime error:");
        std::cout << "Error " << err.m_code << ": " << err.m_message << " (" << err.m_notes << ")" << std::endl;
    } catch(...) {
        dbg_printf("FAIL\n");
        mexErrMsgTxt("Runtime error:");
        std::cout << "Unknown error occurred!" << std::endl;
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
        mexErrMsgTxt("Runtime error:");
        std::cout << "Error " << err.m_code << ": " << err.m_message << " (" << err.m_notes << ")" << std::endl;
        len = -1;
    } catch(...) {
        dbg_printf("FAIL\n");
        mexErrMsgTxt("Runtime error:");
        std::cout << "Unknown error occurred!" << std::endl;
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
        mexErrMsgTxt("Runtime error:");
        std::cout << "Error " << err.m_code << ": " << err.m_message << " (" << err.m_notes << ")" << std::endl;
    } catch(...) {
        dbg_printf("FAIL\n");
        mexErrMsgTxt("Runtime error:");
        std::cout << "Unknown error occurred!" << std::endl;
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
            (*(g_kernels[kernel_idx]))[arg_idx] = g_buffers[buf_idx];
        } else {
            (*(g_kernels[kernel_idx]))(arg_idx, sz ) = pdata;
        }
        return_val = 1;
    } catch(OCLError err) {
        dbg_printf("FAIL\n");
        mexErrMsgTxt("Runtime error:");
        std::cout << "Error " << err.m_code << ": " << err.m_message << " (" << err.m_notes << ")" << std::endl;
    } catch(...) {
        dbg_printf("FAIL\n");
        mexErrMsgTxt("Runtime error:");
        std::cout << "Unknown error occurred!" << std::endl;
    }
    plhs[0] = mxCreateLogicalScalar(return_val);
}
/*
    try {	
		std::vector<cl_platform_id> platforms = OCLPlatform::get_platform_ids();
        
        OCLPlatform p(platforms[platform_idx]); 
        OCLContext context(p);
        
        std::vector<cl_device_id> devices = p.get_device_ids();        
		for (size_t j=0; j<device_idx.size(); ++j) {            
            context += devices[device_idx[j]];            
		}
        
		context.create();
        
		OCLProgram prog(context);

        OCLCommandQueue queue(context, devices[device_idx[0]]); 

		prog.add_source(&kernel_filename[0]);
		prog.create();
		prog.build();        
        
        std::vector<OCLBuffer> buffers;
        
        len = arg_mode.size();
        buffers.resize(len);
        
        for (int i=0; i<len; ++i) {
            int   sz   = arg_size[i];
            char *mode      = &(arg_mode[i][0]);
            char *type      = &(arg_type[i][0]);            
            int flags;
            
            if (mode[0] == 'r') {
                if (mode[1] == 'o') {
                    flags = MEM_FLAGS_READ_ONLY;
                } else if (mode[1] == 'w') {
                    flags = MEM_FLAGS_READ_WRITE;
                }               
            } else if (mode[0] == 'w') {
                if (mode[1] == 'o') {
                    flags = MEM_FLAGS_WRITE_ONLY;
                }                
            } else if (mode[0] == 'l') {
                flags = 0;    
                sz = 0;
            } else if (mode[0] == 't') {
                flags = 0;
                sz = 0;                
            }
            if (mode[0] == 'l'  || mode[0] == 't') {
                //Do nothing.
            } else {
                OCLBuffer b(context, flags, sz);                
                b.create();
                buffers[i] = b;                                
            }
        }
        
        int ndims = 0;
        for (ndims =0; (ndims < 3) && (global_size[ndims] > 0) ; ++ndims) {}
        
        //For now only accept first device:        
        OCLEvent evt;
        
        //1. Queue copy to buffer commands 
        len = arg_mode.size();
        for (int i=0; i<len; ++i) {
            int   sz   = arg_size[i];
            char *mode      = &(arg_mode[i][0]);
            char *type      = &(arg_type[i][0]);                                    
            void *data      = reinterpret_cast<void *>(&(arg_data[i][0]));
            
            if (mode[0] == 'r') {
                queue.enqueue_buffer_copy(buffers[i], data, sz);
            }
        }

        OCLKernel kernel(prog, &kernel_name[0]);
			kernel.set_global_offset(0,0,0);
			kernel.set_ndims(ndims);
			kernel.set_local_size(local_size[0],local_size[1],local_size[2]);
            kernel.set_global_size(global_size[0],global_size[1],global_size[2]);                    
          
        len = arg_mode.size();
        for (int i=0; i<len; ++i) {
            int   sz   = arg_size[i];
            char *mode = &(arg_mode[i][0]);
            void *data = &(arg_data[i][0]);
            
            if (mode[0] == 'l') {
                kernel(i, sz) = 0;
            } else if (mode[0] == 't') {
                kernel(i, sz) = data;
            } else {
                kernel[i] = buffers[i];
            }
        }
        
        //2. Queue the kernel
        dbg_printf("Queue kernel: \n");
        queue.enqueue_ndrange_kernel(kernel);
        
        //3. Queue copying to output buffers
        dbg_printf("Queue copy-from-buffers : \n");
        std::vector<std::vector<char> > out_data;        
        std::vector<char *> fieldname_arr;
        std::vector<char *> typename_arr;        
        
        len = arg_mode.size();        
        for (int i=0; i<len; ++i) {
            int   sz        = arg_size[i];
            char *type      = &(arg_type[i][0]);  
            char *fieldname = &(arg_fieldname[i][0]);
            char *mode      = &(arg_mode[i][0]);
            
            if (fieldname[0] == 0) {
                continue;
            }
            
            if (mode[0] == 'l' || mode[0] == 't') {                
                std::vector<char> data;
                data.resize(sz);
                memcpy(&data[0], &(arg_data[i][0]),sz);
                out_data.push_back(data);
                fieldname_arr.push_back(fieldname);            
                typename_arr.push_back(type);
                continue;
            }
                
            fieldname_arr.push_back(fieldname);            
            typename_arr.push_back(type);
            
            std::vector<char> data;                      
            data.resize(sz);
            out_data.push_back(data);
            
            void *dst      =  reinterpret_cast<void *>(&(out_data[out_data.size()-1][0])); //reinterpret_cast<void*>(&data[0]); 
            queue.enqueue_buffer_copy(dst, buffers[i], sz, 0, CL_FALSE);
        }                                                
        dbg_printf("Waiting for GPGPU to finish...\n");
        queue.finish();                        
        
        //Now copy output:
        
       //4. Now need to interface back to MATLAB
       mwSize dims[2] = {1,1};
       dbg_printf("Copying variables back to matlab...\n");
       const char **ptr_fieldnames = (const char **) (&(fieldname_arr[0]));
       
       dbg_printf("Number of fields = %d\n", fieldname_arr.size());
       for (int i=0; i<fieldname_arr.size(); ++i) {
            dbg_printf("Field %d: %s\n", i, ptr_fieldnames[i]);           
       }
       
       plhs[0] = mxCreateStructArray(2, dims, fieldname_arr.size(), ptr_fieldnames);
       
       mwSize mrows = 1;                              
       len = fieldname_arr.size();
       for (int i=0; i<len; ++i) {
            mwSize ncols = out_data[i].size();
            int    nBytes = ncols;
            int    nElems = nBytes;
            char   *type  = typename_arr[i];
            
            if (strcmp(type, "int8") == 0) {                
                arr = mxCreateNumericMatrix(mrows,nElems, mxINT8_CLASS, mxREAL);
            } else if (strcmp(type, "int16") == 0) {
                nElems /= 2;
                arr = mxCreateNumericMatrix(mrows,nElems, mxINT16_CLASS, mxREAL);                                
            } else if (strcmp(type, "int32") == 0) {
                nElems /= 4;
                arr = mxCreateNumericMatrix(mrows,nElems, mxINT32_CLASS, mxREAL);
            } else if (strcmp(type, "int64") == 0) {
                nElems /= 8;
                arr = mxCreateNumericMatrix(mrows,nElems, mxINT64_CLASS, mxREAL);
            } else if (strcmp(type, "uint8") == 0) {
                arr = mxCreateNumericMatrix(mrows,nElems, mxUINT8_CLASS, mxREAL);
            } else if (strcmp(type, "uint16") == 0) {
                nElems /= 2;
                arr = mxCreateNumericMatrix(mrows,nElems, mxUINT16_CLASS, mxREAL);
            } else if (strcmp(type, "uint32") == 0) {
                nElems /= 4;
                arr = mxCreateNumericMatrix(mrows,nElems, mxUINT32_CLASS, mxREAL);                
            } else if (strcmp(type, "uint64") == 0) {
                nElems /= 8;
                arr = mxCreateNumericMatrix(mrows,nElems, mxUINT64_CLASS, mxREAL);
            } else if (strcmp(type, "single") == 0) {
                nElems /= 4;
                arr = mxCreateNumericMatrix(mrows,nElems, mxSINGLE_CLASS, mxREAL);
            } else if (strcmp(type, "double") == 0) {
                nElems /= 8;
                arr = mxCreateNumericMatrix(mrows,nElems, mxDOUBLE_CLASS, mxREAL);                
            } else if (strcmp(type, "char") == 0) {
                mwSize dims[2] = {1, ncols};                
                arr = mxCreateCharArray(2, dims);
            } else if (strcmp(type, "logical") == 0) {
                mwSize dims[2] = {1, ncols};                
                arr = mxCreateLogicalArray(2, dims);                
            } else {
               //Unsupported!
            }
            
            void *lhs_data = mxGetData(arr);
            memcpy(lhs_data, &(out_data[i][0]), nBytes);
            
            //Now set structure
            mxSetField(plhs[0], 0, fieldname_arr[i], arr);
        }
       dbg_printf("Reached end\n");
       
	} catch (OCLError err) {        
        mexErrMsgTxt("Runtime error:");        
		std::cout << "Error " << err.m_code << ": " << err.m_message << " (" << err.m_notes << ")" << std::endl;
	} catch (...) {
        mexErrMsgTxt("Runtime error:");
		std::cout << "Unknown error occurred!" << std::endl;
	}  
    
    return;
}
*/
