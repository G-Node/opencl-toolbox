% Copyright (C) 2011 by Radford Ray Juang
% clkernel is a class that represents an OpenCL kernel object.
% It is used to create a function out of the CL kernels that were compiled
% and sent to the GPGPU device using the opencl.addfile and opencl.build
% functions. 
% 
% An example usage:
%
% ocl = opencl();
% ocl.initialize();
%
% ocl.addfile('cl/simple_add.cl');
% ocl.build();
%
% k = clkernel('add');
%
%
%

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
                elseif strcmp(S.class, 'clobject')
                    bufferid = argval.buffer.id;
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
