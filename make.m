% %%%%%%%%%%%%%%%% CONFIGURATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%opencl_include_dir = '/usr/include';
%opencl_lib_dir = '/usr/lib';
opencl_include_dir = 'C:\Program Files\ATI Stream\include';
opencl_lib_dir = 'C:\Program Files\ATI Stream\lib\x86';
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mex('src/openclcmd.cpp', '-Iinclude', ['-I' opencl_include_dir], ...
    ['-L' opencl_lib_dir], '-lOpenCL' );
