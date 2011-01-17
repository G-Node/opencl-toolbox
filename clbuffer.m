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
                otherwise,
                    unit_size = 1;
            end
            
            self.id = openclcmd('create_buffer', mode, uint32(unit_size*nelems));
            self.type = type;
            self.num_elems = nelems;
            self.num_bytes = uint32(unit_size*nelems);
            self.mode = mode;            
            self.device = device;            
        end
        
        function data = get(self)
            data = openclcmd('get_buffer', device-1, self.id, self.num_elems, self.type);            
        end
        
        function set(self, data)
            data = feval(self.type, data);
            openclcmd('set_buffer', device-1, self.id, data);
        end
    end
end