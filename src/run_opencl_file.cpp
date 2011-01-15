/* Author: Radford Ray Juang
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
    
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{   
    mxArray *arr    = 0;
    mxArray *cell_arr    = 0;
    size_t   len    = 0;
    std::vector<char> kernel_filename;
    std::vector<char> kernel_name;    
    std::vector<int>  device_idx;
    
    std::vector<std::vector<char> >          arg_mode;
    std::vector<std::vector<char> >          arg_type;
    std::vector<std::vector<char> >          arg_fieldname;
    std::vector<std::vector<unsigned char> > arg_data;    
    std::vector<int>                        arg_size; 
    
    int               platform_idx;
    unsigned int      global_size[3] = {0,0,0};
    unsigned int      local_size[3]  = {0,0,0};    
    unsigned int     *pdata_uint32 = 0;
    
    if (nrhs < 1) {
        mexErrMsgTxt("Insufficient # of arguments");
        return;
    }
   
    //Get filename
    arr = mxGetField(prhs[0],0,"filename");
        len = mxGetNumberOfElements(arr);   
        kernel_filename.resize(len+1);
        mxGetString(arr, &kernel_filename[0], len+1);        
       
    // Get kernel name
    arr = mxGetField(prhs[0],0,"name");
        len = mxGetNumberOfElements(arr);
        kernel_name.resize(len+1);
        mxGetString(arr, &kernel_name[0], len+1);
        
    // Get global size
    arr = mxGetField(prhs[0], 0, "global_size");
        len   = mxGetNumberOfElements(arr);
        pdata_uint32 = reinterpret_cast<unsigned int *>( mxGetData(arr));
        if (len > 3) len = 3;
        for (int i=0; i<len; ++i)            
            global_size[i] = pdata_uint32[i];
        
    // Get local size
	arr = mxGetField(prhs[0], 0, "local_size");
        len   = mxGetNumberOfElements(arr);
        pdata_uint32 = reinterpret_cast<unsigned int *>( mxGetData(arr));
        if (len > 3) len = 3;
        for (int i=0; i<len; ++i)            
            local_size[i] = pdata_uint32[i];
        
    // Get selected platform idx:
    arr = mxGetField(prhs[0], 0, "platform");
        platform_idx = static_cast<int>(mxGetScalar(arr));
        
    // Get selected device(s) idx:
    arr = mxGetField(prhs[0], 0, "devices");		
		len = mxGetNumberOfElements(arr);
		device_idx.resize(len);
        
        pdata_uint32 = reinterpret_cast<unsigned int *>( mxGetData(arr));
        for (int i=0; i<len; ++i)
            device_idx[i] = pdata_uint32[i];
    
    arr = mxGetField(prhs[0], 0, "arg_type");
        len = mxGetNumberOfElements(arr);         
        for (int i=0; i<len; ++i) {
            std::vector<char> str;            
            cell_arr = mxGetCell(arr, i);            
            int cell_len = mxGetNumberOfElements(cell_arr)+1;
            str.resize(cell_len);
            mxGetString(cell_arr, &str[0], cell_len);
            arg_type.push_back(str);            
        }        
        
    arr = mxGetField(prhs[0], 0, "arg_mode");
        len = mxGetNumberOfElements(arr);         
        for (int i=0; i<len; ++i) {
            std::vector<char> str;            
            cell_arr = mxGetCell(arr, i);            
            int cell_len = mxGetNumberOfElements(cell_arr)+1;
            str.resize(cell_len);
            mxGetString(cell_arr, &str[0], cell_len);
            arg_mode.push_back(str);            
        }       
        
    arr = mxGetField(prhs[0], 0, "arg_size");
		len = mxGetNumberOfElements(arr);
        arg_size.resize(len);        
        for (int i=0; i<len; ++i ) {
            cell_arr = mxGetCell(arr, i);            
            arg_size[i] = static_cast<int>(mxGetScalar(cell_arr));
        }        
        
    arr = mxGetField(prhs[0], 0, "arg_fieldname");
        len = mxGetNumberOfElements(arr);         
        for (int i=0; i<len; ++i) {
            std::vector<char> str;            
            cell_arr = mxGetCell(arr, i);            
            int cell_len = mxGetNumberOfElements(cell_arr)+1;
            str.resize(cell_len);
            mxGetString(cell_arr, &str[0], cell_len);
            arg_fieldname.push_back(str);            
        }       
        
    // 
    // Get argument data
    arr = mxGetField(prhs[0], 0, "arg_data");
        len = mxGetNumberOfElements(arr);        
        for (int i=0; i<len; ++i) {
            std::vector<unsigned char> data;
            cell_arr = mxGetCell(arr, i);
            int cell_len = mxGetNumberOfElements(cell_arr) * static_cast<int>(mxGetElementSize(cell_arr));            
            data.resize(cell_len);
            memcpy(&data[0], mxGetData(cell_arr), cell_len);
            arg_data.push_back(data);            
        }       
    #if 0
    // VALIDATION:
        printf("Filename: %s\n", &kernel_filename[0]);
        printf("Kernel name: %s\n", &kernel_name[0]);
        printf("Platform index: %d\n", platform_idx);
        printf("Device index: ");
        for (int i=0; i<device_idx.size(); ++i) {
           printf("%d ", device_idx[i]);           
        }
        printf("\n");
        printf("Global Size: [%d %d %d]\n", global_size[0],global_size[1],global_size[2]);
        printf("Local Size: [%d %d %d]\n", local_size[0],local_size[1],local_size[2]);        
        for (int i=0; i<arg_mode.size(); ++i ) {
            printf("Arg %d : \n", i);
            printf("\t size = %d\n", arg_size[i]);
            printf("\t type = %s\n", &(arg_type[i][0]));
            printf("\t fieldname = %s\n", &(arg_fieldname[i][0]));            
            printf("\t mode = %s\n", &(arg_mode[i][0]));            
            printf("\t data size = %d\n", arg_data[i].size());
        }
    #endif

    dbg_printf("Attempting main algorithm: \n");
    try {	
		std::vector<cl_platform_id> platforms = OCLPlatform::get_platform_ids();
        
        dbg_printf("Establish platform: \n");
        //Establish platform and context
        OCLPlatform p(platforms[platform_idx]); 
        
        dbg_printf("Establish context: \n");
        OCLContext context(p);
        
        dbg_printf("Add devices: \n");
        //Add devices to context:
        std::vector<cl_device_id> devices = p.get_device_ids();        
		for (size_t j=0; j<device_idx.size(); ++j) {            
            context += devices[device_idx[j]];            
		}
        
        dbg_printf("Create context: \n");
        //Create the context
		context.create();
        
        dbg_printf("Setup program: \n");
        //Now setup kernel, etc.
		OCLProgram prog(context);

        dbg_printf("Add sources or binaries: \n");
		//Add sources or binaries
		prog.add_source(&kernel_filename[0]);

		dbg_printf("Create program: \n");
        //Then create the program
		prog.create();

        dbg_printf("Build program: \n");
		//Build the program on specified device (or all devices in context):
		prog.build();        
        
        std::vector<OCLBuffer> buffers;
        
        dbg_printf("Setup buffers: \n");
        //Now need to setup the buffers:                
        len = arg_mode.size();
        buffers.resize(len); 

        for (int i=0; i<len; ++i) {
            dbg_printf("Variable %s\n", &(arg_fieldname[i][0]));            
            int   sz   = arg_size[i];
            dbg_printf("\tSize: %d\n", sz);            
            char *mode      = &(arg_mode[i][0]);
            dbg_printf("\tMode: %s\n", mode);            
            char *type      = &(arg_type[i][0]);            
            dbg_printf("\tType: %s\n", type);            
            int flags;
            
            if (mode[0] == 'r') {
                if (mode[1] == 'o') {
                    flags = MEM_FLAGS_READ_ONLY;
                } else if (mode[1] == 'w') {
                    flags = MEM_FLAGS_READ_WRITE;
                } else {
                    
                }               
            } else if (mode[0] == 'w') {
                if (mode[1] == 'o') {
                    flags = MEM_FLAGS_WRITE_ONLY;
                } else {
                    
                }                
            } else if (mode[0] == 'l') {
                flags = 0;    
                sz = 0;
            } else if (mode[0] == 't') {
                flags = 0;
                sz = 0;                
            }
            dbg_printf("\tFlags: %d\n", flags);
            dbg_printf("\tUpdated sz: %d\n", sz);
            if (mode[0] == 'l'  || mode[0] == 't') {
                //Do nothing.
            } else {
                dbg_printf("\tMaking buffer: \n");
                OCLBuffer b(context, flags, sz);

                dbg_printf("\tCalling buffer create: \n");
                b.create();

                dbg_printf("\tCopying buffer to stack: \n");
                buffers[i] = b;
            }
            dbg_printf("\tDone\n");
        }
        
        int ndims = 0;
        for (ndims =0; (ndims < 3) && (global_size[ndims] > 0) ; ++ndims) {}
        
        dbg_printf("Setup command queue: \n");
        //For now only accept first device:        
        OCLCommandQueue queue(context, devices[device_idx[0]]); 
        OCLEvent evt;
        
        dbg_printf("Setup kernel: \n");
        OCLKernel kernel(prog, &kernel_name[0]);
			kernel.set_global_offset(0,0,0);
			kernel.set_ndims(ndims);
			kernel.set_local_size(local_size[0],local_size[1],local_size[2]);
            kernel.set_global_size(global_size[0],global_size[1],global_size[2]);                    
          
        dbg_printf("Setup kernel args: \n");
        //Setup kernel arguments
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
                
        dbg_printf("Queue copy-to-buffer commands: \n");
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
            dbg_printf("Filling in output %d\n", i);
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
            
            dbg_printf("# elems =  %d\n", nElems);
            dbg_printf("# bytes =  %d\n", nBytes);

            void *lhs_data = mxGetData(arr);
            memcpy(lhs_data, &(out_data[i][0]), nBytes);
            
            dbg_printf("Setting Field\n");
            //Now set structure
            mxSetField(plhs[0], 0, fieldname_arr[i], arr);
        }
      
       dbg_printf("Reached end\n");
      
       //Clean out buffers:
       /*
       for (int j=0; j<buffers.size(); ++j) {
            delete buffers[j];
            buffers[j] = 0;
       }
       buffers.clear();
       */
	} catch (OCLError err) {        
        mexErrMsgTxt("Runtime error:");        
		std::cout << "Error " << err.m_code << ": " << err.m_message << " (" << err.m_notes << ")" << std::endl;
	} catch (...) {
        mexErrMsgTxt("Runtime error:");
		std::cout << "Unknown error occurred!" << std::endl;
	}  
    
    return;
}

