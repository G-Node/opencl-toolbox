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
            data = [];

            if self.id >= 0, 
                data = openclcmd('get_buffer', self.device-1, self.id, self.num_elems, self.type);            
            end
        end
        
        function set(self, data)
            if self.id < 0,
                return;
            end

            data = feval(self.type, data);
            openclcmd('set_buffer', self.device-1, self.id, data);
        end
        
        function delete(self)
            openclcmd('destroy_buffer', self.id);
        end
    end
end
