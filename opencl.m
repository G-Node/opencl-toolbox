% This class interfaces with the OpenCL framework
%
% Author: Radford Ray Juang (January 2011)
%   (rayver _/at\_ hkn (dot) berkeley (dot) edu )
%
classdef opencl < handle
    properties (GetAccess = public, SetAccess = protected)
        platforms = {}        
        
        selected_platform = 1;  
        selected_device   = 1;
        cl_filename       = [];        
    end
    
    methods
        
        function this = opencl()
        % Creates an OpenCL object with retrieved information about
        % availble platform and indices.
        %        
            this.platforms = get_opencl_info();            
        end
        
        function set_platform(this, id) 
        % set_platform(OBJ, PID)
        %
        % Sets the platform to use. This should correspond to
        % platforms(PID) platform.
        %
            this.selected_platform = id;            
        end
        
        function set_devices(this, ids)
        % set_devices(OBJ, IDS)
        % 
        % Sets the devices to load. These should correspond to the 
        % device index in platforms{PID}.devices{IDS}
        %             
            this.selected_device = ids;            
        end
        
        function load(this, cl_filename) 
        % load(OBJ, FILENAME)
        % 
        % Sets the OpenCL file to load
        %         
            this.cl_filename = cl_filename;
        end
                
        function output = run(this, kernel_name, kernel_args, global_dimensions, local_dimensions)
        % OUTPUT = run(NAME, ARGS)
        % OUTPUT = run(NAME, ARGS, GLOBAL_DIMENSIONS)
        % OUTPUT = run(NAME, ARGS, GLOBAL_DIMENSIONS, LOCAL_DIMENSIONS)
        %
        % NAME is a string specifying the kernel to run
        % ARGS is a Nx4 cell array where the each row
        %   specifies an argument. Each row contains
        %   the following items:
        %      [data, permission, bytesize, output_fieldname]
        % 
        %   data       is the data values in the correct type size.
        %              For outputs, must specify at least 1 value of the
        %              correct type. 
        %
        %   permission is the permission (ro, wo, rw, l, or t) to assign to 
        %              the buffer. 
        %                ro  = read only memory buffer
        %                wo  = write only memory buffer
        %                rw  = read write memory buffer
        %                l   = local memory space (no memory buffer)
        %                t   = variable type (no memory buffer) 
        %
        %   bytesize   is the size to allocate to the buffer. If left
        %              empty, then it is set to numel(data) x
        %              sizeof(type(data))
        %
        %   output_fieldname  is a string specifying the fieldname of the
        %              output to read the buffer contents to. If [], then
        %              this variable is not read out.
        %        
        %
        % GLOBAL_DIMENSIONS is an array of up to 3 values containing the
        %  global dimension (number of threads to break the problem into)
        % 
        % LOCAL_DIMENSIONS is an array of up to 3 values containing the
        %  local dimension (or number of threads to allow sharing memory 
        %  locally as a workgroup)
        % 
        % OUTPUT  will contain the following fields:
        %    .(fieldname) -- provided fieldname containing output values
        %        
            args.filename      = this.cl_filename;
            args.name          = kernel_name;
            args.platform      = uint32(this.selected_platform -1);
            args.devices       = uint32(this.selected_device -1);
            
            nArgs              = size(kernel_args,1);
            args.arg_data      = cell(nArgs, 1);
            args.arg_type      = cell(nArgs, 1);
            args.arg_mode      = cell(nArgs, 1);
            args.arg_size      = cell(nArgs, 1);
            args.arg_fieldname = cell(nArgs, 1);
            args.global_size   = uint32(global_dimensions);
            args.local_size    = uint32(local_dimensions);
            
            for i=1:nArgs,
               args.arg_data{i}      = kernel_args{i,1};
               args.arg_type{i}      = class(kernel_args{i,1});
               args.arg_mode{i}      = kernel_args{i,2};
               args.arg_size{i}      = uint32(kernel_args{i,3});
               args.arg_fieldname{i} = kernel_args{i,4};
               
               if isempty(kernel_args{i,3}),
                   dataval_ = args.arg_data{i};  % Used to get the size in bytes
                   s = whos('dataval_');
                   args.arg_size{i} = uint32(s.bytes);
               end
            end
            
            % Run CL file:
            output = run_opencl_file(args);
        end
    end           
end
    
