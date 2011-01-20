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
    end
end
