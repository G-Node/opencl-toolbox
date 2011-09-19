% %%%%%%%%%%%%%%%% CONFIGURATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%opencl_include_dir = '/usr/include';
%opencl_lib_dir = '/usr/lib';

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

arch = computer('arch');

opencl_include_dir = '/usr/include';
cxxflags = '';
ldflags = '';
libs = ' ';

switch arch
  case {'win32', 'win64'}
    % NVidia or ATI SDKs could be installed anywhere, lets ask
    prompt = 'Please enter the path where to find the OpenCL ';
    
    opencl_include_dir = input ([prompt 'include directory'], 's');
    opencl_lib_dir = input ([prompt 'library directory'], 's');
    libs = ['-L' opencl_lib_dir ' -lOpenCL'];
    
  case {'maci64','maci32'}
    opencl_include_dir = '/System/Library/Frameworks/OpenCL.framework/Headers';
    cflags = '-framework OpenCL';
    ldflags = '-framework OpenCL';
    
  case {'glnx86'}
    libs = '-lOpenCL';
    cxxflags = '-m32';

end

ldflags = ['LDFLAGS=\$LDFLAGS ' ldflags];
cxxflags = ['CXXFLAGS=\$CXXFLAGS ' cxxflags ];

mex('-v', 'src/openclcmd.cpp', '-Iinclude', ['-I' opencl_include_dir], ...
    libs, cxxflags, ldflags);

