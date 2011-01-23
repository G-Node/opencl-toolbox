% clbuffer is an encapsulation of an OpenCL buffer object. 
% Buffers are associated with a specific device and represents
% memory available for a variable to hold its value.
% 
% To allocate a clBuffer, you need to know the type and the 
% number of elements in the type that you are going to store.
% For example, a vector of N doubles would be initialized as follows:
%
%   bufA = clbuffer('rw', 'double', N, deviceIdx)
% 
% where deviceIdx can be omitted (and would default to 1). deviceIdx specifies
% the index of the device in the device id array that was passed in the call
% of opencl.initialize( ) 
% 
% To view values inside a buffer:
%   values = bufA.get()
%
% To set values inside a buffer:
%   bufA.set(values)
%
% It is important to note that the get/set operations are blocking.
%
% Finally, to free a buffer:
%   clear bufA;
%
% See also: clbuffer/clbuffer
%           clbuffer/get
%           clbuffer/set
%           clbuffer/delete
%
% Author: Radford Ray Juang


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
%
classdef clbuffer < handle
    properties(GetAccess = public, SetAccess = protected)
        id = [];
        device = [];
        num_bytes = [];
        num_elems = [];
        type = [];
        mode = [];
    end
    
    methods
        function self = clbuffer(mode, type, nelems, device)
        %  clbuffer(mode, type, nelems)
        %  clbuffer(mode, type, nelems, device)
        %
        %  Create a buffer of the given specifications:
        %   mode :  can be 'ro' (read-only), 
        %                  'wo' (write-only),
        %                  'rw' (read-write)
        %   type :  can be 'int64', 'uint64', 'double', 
        %                  'int32', 'uint32', 'single',
        %                  'int16', 'uint16', 
        %                  'int8', 'uint8', 'char', 'logical',
        %                  'local'
        %           'local' means the buffer is cache space on the device
        %            to be shared between local workgroups in a compute unit
        %
        %  nelems : number of elements of the specified type. for 'local',
        %           this is the number of bytes to reserve for the local
        %           cache.
        %  
        %  device : (default 1) index of the device initialized to create the
        %           buffer for
        %
            if nargin < 4,
                device = [];
            end
            
            if isempty(device),
                device = 1;
            end
            
            unit_size = 1;
            switch type,
                case {'int64', 'uint64', 'double'},
                    unit_size = 8;
                case {'int32', 'uint32', 'single'},
                    unit_size = 4;
                case {'int16', 'uint16'},
                    unit_size = 2;
                case {'int8', 'uint8', 'char', 'logical'},
                    unit_size = 1;
                case {'local'},
                    unit_size = 0;
                otherwise,
                    unit_size = 1;
            end

            self.id = -1;
            if unit_size > 0, 
                self.id = openclcmd('create_buffer', mode, uint32(unit_size*nelems));
            else
                unit_size = 1; % Set to 1 so that we can properly compute the bytes
            end

            self.type = type;
            self.num_elems = nelems;
            self.num_bytes = uint32(unit_size*nelems);
            self.mode = mode;            
            self.device = device;            
        end
        
        function data = get(self)
        % obj.get()
        %
        % Fetches the memory contents of the buffer from device memory to host
        % memory. This call is blocking and returns with the values of the
        % buffer in the device.
        %
            data = [];

            if self.id >= 0, 
                data = openclcmd('get_buffer', self.device-1, self.id, self.num_elems, self.type);            
            end
        end
        
        function set(self, data)
        % obj.set(data) 
        % 
        % Copies the contents of data from host memory to device memory. This
        % call is blocking and returns after the contents have been fully
        % transfered to the device. Note: it is important to have the correct
        % number of elements in data. The values in data are automatically
        % cast to the correct type. (so you can pass a double array here and
        % it will be converted to whichever type was specified when the buffer
        % was created)
        %
            if self.id < 0,
                return;
            end

            data = feval(self.type, data);
            openclcmd('set_buffer', self.device-1, self.id, data);
        end
        
        function delete(self)
        % delete(obj)
        % 
        % Deletes the specified object and frees it from memory. If already
        % freed, the command just returns.
        %
            openclcmd('destroy_buffer', self.id);
        end
    end
end
