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
    end
    
    methods
        
        function this = opencl()
        % Creates an OpenCL object with retrieved information about
        % availble platform and indices.
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
        % (where first index is 1). If unspecified, the first device is 
        % used. 
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
            
            platform = uint32(platform - 1);
            devices = uint32(devices - 1);
            
            result = openclcmd('initialize', platform, devices);
            if ~result,
                error('OpenCL platform and device could not be initialized.');
            end
            this.selected_platform = platform;
            this.selected_device = devices;
        end
        
    end           
end
    
