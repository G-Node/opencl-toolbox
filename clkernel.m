classdef clkernel < handle
    properties 
        device = 1
        id = [];        
    end
    
    methods 
        function self = clkernel(kernelname, target_device, global_dim, local_dim)
            if nargin < 3,
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
                value = self.execute(index.subs{:});
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
                if strcmp(S.class, 'clbuffer'),
                    % If yes, then set kernel arg to the clbuffer id        
                    %openclcmd('set_kernel_args',  kernel_id, arg_num, buffer_id, [], 0 )
                    buffer_id = argval.id;
                    if buffer_id >= 0,
                        openclcmd('set_kernel_args',  self.id, argnum, buffer_id, [], 0 );
                    else
                        %Setting kernel argument to local variable type:
                        %  set_kernel_args( kernel_id, arg_num, -1, [], nBytes )          
                        nBytes = argval.num_bytes;
                        openclcmd('set_kernel_args', self.id, argnum, -1, [], nBytes);
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
                                       
                    openclcmd('set_kernel_args', self.id, argnum, -1, argval, 0);
                else
                    error('Invalid type');
                end                
            end
            
            openclcmd('execute_kernel', self.device, self.id);
        end        
    end
end
