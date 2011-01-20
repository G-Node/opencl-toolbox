classdef clkernel < handle
    properties 
        device = 1
        id = [];        
    end
    
    methods 
        function self = clkernel(kernelname, global_dim, local_dim, target_device)

            if nargin < 4,
                target_device = [];
            end

            if isempty(target_device),
                target_device = 1;
            end
            
            self.device = target_device;               
            self.id = openclcmd('create_kernel', uint32(global_dim), uint32(local_dim), kernelname);
            
            % TODO: Method to automatically determine optimal global and local dimensions 
            
        end

        function value = subsref(self, S)
            % Make sure we allow for subsequent calls            
            index = S(1);
            if strcmp(index.type, '.'),
                % Function call or variable name?
                if ismethod(self, index.subs), 
                    % Function call
                    if numel(S) > 1 && strcmp(S(2).type, '()'),
                        args = S(2);
                        value = feval(index.subs, self, args.subs{:});
                        S(1:2) = [];
                    else
                        value = feval(index.subs, self);
                        S(1) = [];                        
                    end
                else
                    % Variable
                    S(1) = [];
                    value = self.(index.subs);
                end
            else
                self.execute(index.subs{:});
                value = [];
                S = S(2:end);
            end
            
            if ~isempty(S),
                value = subsref(value, S);
            end
        end
        
        function execute(self, varargin)            
            for i=1:numel(varargin) 
                argnum = i-1;
                argval = varargin{i};

                % Is the argument a clbuffer?
                S = whos('argval');
                
                kernelid = self.id;
                bufferid = -1;
                data = [];
                nbytes = 0;
                
                if strcmp(S.class, 'clbuffer'),
                    % It can be a buffer with actual data or buffer that is
                    % empty.
                    %                    
                    bufferid = argval.id;
                    if bufferid < 0,
                        %Local variable type:                        
                        nbytes = argval.num_bytes;
                    end
                    
                elseif strcmp(S.class, 'double') || ...
                       strcmp(S.class, 'single') || ...
                       strcmp(S.class, 'char') || ...
                       strcmp(S.class, 'uint8') || ...
                       strcmp(S.class, 'uint16') || ...
                       strcmp(S.class, 'uint32') || ...
                       strcmp(S.class, 'uint64') || ...
                       strcmp(S.class, 'int8') || ...
                       strcmp(S.class, 'int16') || ...
                       strcmp(S.class, 'int32') || ...
                       strcmp(S.class, 'logical')
                    data = argval;
                else
                    error('Invalid type');
                end 
                
                openclcmd('set_kernel_args', kernelid, argnum, bufferid, data, int32(nbytes));
                %fprintf(1, 'set_kernel_args: kernelid = %d, argnum = %d, buffer=%d, data=%g, sz=%d\n', ...
                %    kernelid, argnum, bufferid, data, nbytes);
            end % for i
            
            openclcmd('execute_kernel', self.device-1, self.id);
        end        
    end
end
