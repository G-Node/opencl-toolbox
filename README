==========[ OpenCL Toolbox ]============================
Version 0.15
Date: 01.24.2011
Website: http://code.google.com/p/opencl-toolbox
SVN: http://opencl-toolbox.googlecode.com/svn/trunk/

==========[ What's New ] ===============================
0.17:  Added additional device information.
       Bug fix: I had swapped the local and global dims.. (big oops). 
       Started adding matlab kernels

0.15:  Updated clobject interface. 
       Additional bug fixes noted and fixed.
       Added destroy_buffer to openclcmd and delete( )
          to matlab interface classes.
       
0.13:  Bug fix: Fixed issue with create_buffer on 
       cpu devices

0.11:  Initial release

==========[ Requirements ] =============================

- OpenCL 1.1+ device driver platform
 (e.g. you can download and install one from:
     http://developer.amd.com/zones/OpenCLZone            
     http://software.intel.com/en-us/articles/intel-opencl-sdk   
     http://www.nvidia.com/opencl
  )
- MATLAB R2009 or later (I've only tested this on R2010)
- C++ compiler (such as g++ or Visual Studio )

==========[ Installation ] =============================

1. In MATLAB, open the make.m file and edit the opencl_include_dir
and opencl_lib_dir to point to the proper folders. 

opencl_include_dir: is the directory containing CL/cl.h
  (e.g. /usr/include  , or C:\Program Files\ATI Stream\include)

opencl_lib_dir: is the directory containing libOpenCL.so (or OpenCL.lib in Windows)
  (e.g. /usr/lib   , or C:\Program Files\ATI Stream\lib\x86 )

2. Execute make 
    make

3. Add this folder to your toolbox path. Now we're ready to go!


========== [ USAGE ] ================================

see test_openclmatlab.m  for examples
or if you want to use the raw interface, see test_openclcmd.m

==========[ TROUBLESHOOTING ]========================
Under Linux, if you get the message like: 

   bin/glnxa64/../../sys/os/glnxa64/libstdc++.so.6: version `GLIBCXX_3.4.11' not found (required by ... 

This can be fixed as follows:

1. Assuming your matlab install is in /sw/matlab/r2010b/ 
   
   cd /sw/matlab/r2010b/sys/os/glnxa64

(on your system, it may be glnx86... depending on which CPU architecture you are using)

2. Remove the symlink libstdc++.so.6 and move the outdated libraries into another folder
   rm libstdc++.so.6
   mkdir outdated
   mv libstdc++.so.* outdated
   mv libgcc_s.so.1 outdated

3. Restart matlab and rerun make.



==========[ To Do list ] =============================

- Better documentation and matlab examples
- Override matlab plus, minus, rdivide, ldivide, times, pow, exp, etc.
- BLAS functionality
- Automatically determining global and local dimensions
- Testing on various platforms

