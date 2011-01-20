classdef clobject < handle
    properties (GetAccess = public, SetAccess = public)
        datatype = 'object';    % String containing the datatype
        dims = [];              % Contains dimension of the data
        device_id = [];         % Contains the device id of the 
        buffer_id = [];         % Contains the buffer id of the 
    end

    methods
        function this = clobject( )
            dims = 0;
        end
       
        function data = fetch(this)
            data = openclcmd('get_buffer', );
            data = reshape(dims);
        end

    end
end
