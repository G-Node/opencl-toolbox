find_package(OpenCL REQUIRED)
find_package(Matlab REQUIRED)
find_program(MEX_COMPILER mex REQUIRED)
find_program(MEX_MEXEXT mexext REQUIRED)

execute_process(
	COMMAND ${MEX_MEXEXT} 
	OUTPUT_VARIABLE mex_extension
        ERROR_VARIABLE mex_extension
	OUTPUT_STRIP_TRAILING_WHITESPACE
	ERROR_STRIP_TRAILING_WHITESPACE
	)
message( "Mex extension assigned: .${mex_extension} " )

include_directories(${OPENCL_INCLUDE_DIRS})
include_directories(${MATLAB_INCLUDE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/include)

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

add_library(openclcmd MODULE openclcmd.cpp)
set_target_properties(openclcmd PROPERTIES PREFIX "")
set_target_properties(openclcmd PROPERTIES SUFFIX ".${mex_extension}")
target_link_libraries(openclcmd ${MATLAB_LIBRARIES} ${OPENCL_LIBRARIES})

