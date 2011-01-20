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
            data = this.buffer.get();
            data = reshape(data, this.dims);
        end

        function set(this, data)

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

    end
end
