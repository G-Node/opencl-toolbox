% opencl class creates a basic interface to the OpenCL device + platform
% This class is responsible for setting up the device/platform to use,
% for adding and building the OpenCL source files, and for forcing a 
% "wait" to occur to allow all operations pending on the GPGPU to finish.
%
% Note: You may only create one instance of this class. Multiple instances
% of this class share the same device state. So, don't attempt to utilize
% multiple instances of this class. It doesn't provide any value.
% 
% All cleanup and device release calls are handled automatically when you
% quit MATLAB or call "clear all"
%
%
% Please refer to the member functions for additional details:
%   opencl/opencl
%   opencl/initialize
%   opencl/addfile
%   opencl/build
%   opencl/wait
%
% Author: Radford Ray Juang
%
% Example usage:
%   ocl = opencl();  % Fetch a list of available platforms and devices
%
%   disp(ocl);  % Inspect the platform and device you want to use
%               % All platforms available are in the member attribute
%               % .platforms )
%               
%   disp(ocl.platforms(1) );
%
%   % Within each platform, there there is a list of devices associated with
%   % the platform. You can view this in the .devices attribute for each
%   % platform
%   
%   disp(ocl.platforms(1).devices);
%   disp(ocl.platforms(1).devices(1));
%
%   % Assuming we want to pick platform 1 and device 1 (as viewed above):
%
%   platform = 1; 
%   device = 1;
%   ocl.initialize(platform, device);
%
%   % Now let's load the kernel cl/simple_add.cl
%   ocl.addfile('cl/simple_add.cl');
%  
%   % You can add additional files you want to use
%   % ocl.addfile('cl/another_file.cl');
%   % ocl.addfile('cl/another_file2.cl');
%
%   ocl.build();   % Now let's build our kernel and send to the GPU device
%  


% Copyright (C) 2011 by Radford Ray Juang
% 
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in
% all copies or substantial portions of the Software.
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
% THE SOFTWARE.
classdef opencl < handle
    properties (GetAccess = public, SetAccess = protected)
        platforms = {}        
        
        selected_platform = 1;  
        selected_device   = 1;
        files_loaded = {};
        built = 0;        
    end
    
    methods
        
        function this = opencl()
        % opencl()
        %
        % Creates an OpenCL object with retrieved information about
        % availble platform and indices. Calling this does not affect 
	    % the previous state of the OpenCL platform.
        %        
	    % For example:
	    %   ocl = opencl();
	    % 
	    % The member attribute .platforms is set to contain information
	    % about the available platform. Example:
	    % 
	    %   disp(ocl.platforms) 
	    %   disp(ocl.platforms(1))
        %   disp(ocl.platforms(1).devices)
        %	
        % FAQ: 
        % Q: What is a platform? 
        % A: A platform is simply the library that implements the 
        % OpenCL functionality. It is installed when you install the
        % OpenCL device driver from the vendor. Each platform can
        % communicate with one or more devices. For example, common
        % platforms are:
        %   - the AMD/ATI Stream platform (which can talk to CPUs and ATI
        %     GPGPUs)
        %   - the NVIDIA platform (which can talk with NVIDIA GPGPUs)
        %   - the Intel platform (which can talk to CPUs for now)
        %
        % Each platform is associated with one or more devices. The device
        % is the physical hardware that contains the GPU/CPU computing device
        % we want to use.
        %
            this.platforms = openclcmd();            
        end
        
        function initialize(this, platform, devices)
        % initialize(obj)
        % initialize(obj, PLATFORM)        
        % initialize(obj, PLATFORM, DEVICES)
        % 
        % Initialize OpenCL interface to use the specified platform and 
        % devices. 
        %
        % PLATFORM is the index of the platform in obj.platforms to
        % use (where first index is 1). If unspecified, 1 is the default 
        % value and the first platform is used.
        %
        % DEVICES is the indices of the devices in the platform to use 
        % (where first index is 1). If unspecified, the first device on
        % the selected PLATFORM is used. 
        %
    	% Note: Calling this function wipes out the previous state of the
	    % interface mex and resets the GPGPU state.
    	% 
    	% Example:
    	% 
    	%   ocl = opencl();
    	%   ocl.initialize(1,2)
    	% 
    	% Set OpenCL to use platform 1 and device 2. Platform 1 information
    	% is available in the member attribute: ocl.platforms(1)
    	% and device 2 is available in the member attribute: ocl.platforms(1).devices(2)
    	%
            if nargin < 2,
                platform = [];
            end
            
            if nargin < 3, 
                devices = [];                
            end
            
            if isempty(platform),
                platform = 1;
            end
            
            if isempty(devices),
                devices = 1;
            end
           
            result = openclcmd('initialize', uint32(platform-1), uint32(devices-1));
            
            if ~result,
                error('OpenCL platform and device could not be initialized.');
            end
            
            this.selected_platform = platform;
            this.selected_device = devices;
        end
        
        function addfile(this, filename)
        % addfile(obj, filename)
        % 
        % Includes the given cl file to be built and sent to the GPGPU.
        % This function only compiles a list of source files we want to
        % include. Compilation and sending the program to the device does
        % not occur until build is called.
        %
        % Example:
	    % 
	    %  ocl = opencl();
	    %  ocl.initialize(1,1);
	    %  ocl.addfile('cl/simple_add.cl');
	    %  ocl.addfile('cl/another_file.cl');
	    %  ocl.addfile('cl/another_file2.cl');
	    %  ocl.build();
	    %
	    % See also opencl/build

            this.files_loaded{end+1} = filename;
            openclcmd('addfile', filename);
        end
        
        function build(this)
        % build(obj)
        % 
        % Build the opencl files and send to GPGPU.
        % Note: Only opencl files added with addfile are compiled
	    % and sent to the GPGPU
	    %
	    % See also opencl/addfile

            openclcmd('build');
            this.built = 1; 
        end
        
        function wait(this, device_id)
        % wait(obj)
        % wait(obj, device)
        % 
        % Waits for device with the given index (first index is 1) 
        % to complete all execution and memory operations.
        %
	    % For example:
	    %   ocl = opencl();
	    %   ocl.initialize(1,[1,3])
	    %     ...
	    %   ocl.wait(2)   
	    % 
	    % This example initializes platform 1 and devices 1 and 3.
	    % Then the wait command forces a wait on the second device specified
	    % (device 3).
	    %
	    %   ocl.wait(1)  
	    % 
	    % will cause a wait until the first device (device 1) finishes all
	    % execution and memory transfer operations.
	    %

            if nargin < 2,
                device_id = [];
            end
            
            if isempty(device_id),
                device_id = 1;
            end
            
            openclcmd('wait_queue', device_id-1);
        end
    end           
end
    
