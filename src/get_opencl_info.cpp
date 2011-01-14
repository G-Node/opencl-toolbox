/*
 * Author: Radford Ray Juang
 * Email: rayver /_at_/ hkn (dot) berkeley (dot) edu
 */

#include <ray/opencl/opencl.h>
#include <vector>
#include <iostream>
#include "mex.h"

using namespace ray::opencl;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
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

