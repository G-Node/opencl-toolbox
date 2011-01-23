% clobject is an encapsulation of an OpenCL object and is used to convert a
% matlab object into a device object. For example:
%
% Given a matlab object:
%   arr = single(1:10);
%   buffA = clobject(arr);
%
% arr is now in device memory and the resulting storage buffer is in bufA.
%
% See clobject/clobject
%     clobject/set
%     clobject/get
%     clobject/delete

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
classdef clobject < handle
    properties (GetAccess = public, SetAccess = public)
        datatype = [];    % String containing the datatype
        dims = [];        % Contains dimension of the data
        device_id = [];   % Contains the device id of the object
        buffer = [];      % Contains the buffer object
    end

    properties (GetAccess = private, SetAccess = private)
        % Valid types
        valid_types = {'int8', 'int16', 'int32', 'int64', ...
                    'uint8', 'uint16', 'uint32', 'uint64', ...
                    'double', 'single', 'char', 'logical' };
    end

    methods
        function this = clobject(data, deviceid)
        % clobject(data)
        % clobject(data, device)
        %
        % Transfer data to device memory and create a clobject 
        % representation for the data. device is the index of the device where
        % the data is stored. If unspecified, it defaults to 1.
        %
            this.dims = size(data);
            if nargin < 2,
                deviceid = [];
            end

            if isempty(deviceid),
                deviceid = 1;
            end

            this.device_id = deviceid;
            S = whos('data');
            this.datatype = S.class;

            if ~ismember(S.class, this.valid_types),
                error('Type of data unsupported!');
            end

            % Create buffer with provided data:
            this.buffer = clbuffer('rw', this.datatype, numel(data), deviceid);
            this.buffer.set(data(:));
        end
       
        function data = get(this)
        % data = obj.get()
        %
        % Copy device memory in obj to host memory
        %
            data = this.buffer.get();
            data = reshape(data, this.dims);
        end

        function set(this, data)
        % obj.set(data)
        % 
        % Copy data in host memory to device memory. 
        %
            S = whos('data');
            if (prod(this.dims) ~= numel(data)) || ...
               (~strcmp(this.datatype, S.class)),

                if ~ismember(S.class, this.valid_types),
                    error('Type of data unsupported!');
                end

                % Need to create new object
                this.datatype = S.class;
                this.buffer = clbuffer('rw', this.datatype, numel(data), this.device_id);
            end

            this.dims = size(data);
            this.buffer.set(data(:));
        end

        function delete(this)
        % delete(obj)
        % 
        % delete the object and free all resources
        %
            delete(this.buffer);
        end

    end
end
