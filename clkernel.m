% clkernel is a class that represents an OpenCL kernel object.
% It is used to create a function out of the CL kernels that were compiled
% and sent to the GPGPU device using the opencl.addfile and opencl.build
% functions. 
% 
% An example usage:
%
%   ocl = opencl();
%   ocl.initialize();
%
%   ocl.addfile('cl/simple_add.cl');
%   ocl.build();
%
% % Create some data objects:
%
%   x = clobject(single(1:10));
%   y = clobject(single(11:20));
%   z = clobject(zeros(1,10, 'single'));
%
%   % z = x+y:
%   global_work_size = [10,0,0];
%   local_work_size = [10,0,0];
%
%   addkernel = clkernel('add', global_work_size, local_work_size);
%   addkernel(x,y,z, uint32(10));
%   
% % Fetch z values:
%   values = z.get();
%
% See clkernel/clkernel 
%     clkernel/subsref
%     clkernel/execute
%
% Author:Radford Ray Juang
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
classdef clkernel < handle
    properties 
        device = 1
        id = [];        
    end
    
    methods 
        function self = clkernel(kernelname, global_dim, local_dim, target_device)
        % obj = clkernel(kernel_name)
        % obj = clkernel(kernel_name, global_work_size)
        % obj = clkernel(kernel_name, global_work_size, local_work_size)
        % obj = clkernel(kernel_name, global_work_size, local_work_size,
        %       target_device)
        %
        % Creates a kernel object that represents the compiled kernel
        % specified by kernel_name. This is the actual __kernel function
        % defined in the cl source files added using opencl.addfile and
        % opencl.build.
        %
        % kernel_name is a string containing the kernel function name in the
        % CL file.
        % 
        % global_work_size is the number of global compute units you want to
        % use. This is a 1x3 array containing the number of compute units in
        % each direction. For example, if you have 256 processors, you can
        % have the processors divide up the workload into 4x4x16 blocks:
        %    global_work_size = [4,4,16]
        % Or, you can divide the data into 16x16 blocks:        %
        %    global_work_size = [16,16,0]
        % Or you can just divide the data into 256 blocks:
        %    global_work_size = [256,0,0]
        %
        % If this is unspecified, the default is to divide into 128 blocks
        %
        % local_work_size specifies the number of local work groups to 
        % divide each global compute unit into. Think of this as like threads
        % on a processor. The global_work_size specifies the number of
        % processors to spread the division of labor over, whereas the
        % local_work_size specifies the number of threads to divide the
        % work for each processor. Threads can communicate and share memory 
        % with one another where as global compute blocks cannot, and it is
        % possible for multiple threads within each global compute unit to
        % execute at a time.
        %
        % Again, if this is unspecified, the default is 128.
        %
        % target_device is the index of the device to execute the kernel on.
        % If you initialized one device, you can safely ignore this parameter. 
        % However, if you've initialized more than one device, e.g.:
        %   ocl=opencl();
        %   ocl.initialize(1,[1,3,5]);
        %
        % then target_device=1 will execute the kernel on device 1
        %      target_device=2 will execute the kernel on device 3
        %      target_device=3 will execute the kernel on device 5
        %
        % If unspecified, the first device index is used. 
        % NOTE: Use of multiple target devices has not been tested.
        %
        % Once a kernel has been created with say:
        %   addkernel = clkernel('add', global_work_size, local_work_size);
        %
        % One can just execute the kernel by using the defined kernel as a
        % function. For example,
        %
        %    addkernel(buffA, buffB, buffC, uint32(10));
        %
        % Constants must be casted to the correct type that the kernel
        % requires. Non-constant variables must be clbuffer or clobject
        % instances. 
        %
        % NOTE: kernel execution is non-blocking. So, the function will 
        % return regardless of if kernel execution is completed.
        %
            if nargin < 2,
                global_dim = [];                
            end

            if nargin < 3,
                local_dim = [];
            end

            if nargin < 4,
                target_device = [];
            end

            if isempty(target_device),
                target_device = 1;
            end
           
            % Automatically pick a size (this is a bad idea in general)
            if isempty(global_dim),
                global_dim = [128, 0,0];
            end
            if isempty(local_dim),
                local_dim = [128,0,0];
            end

            self.device = target_device;               
            self.id = openclcmd('create_kernel', uint32(local_dim), uint32(global_dim), kernelname);
        end

        function value = subsref(self, S)
            % Overrides matlab ( ) functionality and passes the call to the
            % execute function. For example, if a kernel is created as
            % follows:
            %    f = clkernel(kernelname, global_dims, local_dims);
            %
            % Place the execution of the kernel on the device queue by :
            %    f(arg1, arg2, arg3);
            % 
            % And to ensure the execution is complete, make sure you call
            % opencl.wait. Example:
            %
            %   ocl = opencl();
            %   ocl.initialize(1,1);
            %
            %      ... 
            %
            %   ocl.wait();
            %
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
            % obj.execute(arg1, arg2, ...)
            %
            % Place the execution of the kernel on the device queue
            % with the provided arguments.
            % arg1, ... are arguments to the kernel. Constant arguments must
            % be cast to the correct variable type before being passed.
            %
            % Non-constant arguments must be of type clbuffer or clobject
            %
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
