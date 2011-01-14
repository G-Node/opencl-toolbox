#ifndef _RAY_OPENCL_OCLEVENT_H_
#define _RAY_OPENCL_OCLEVENT_H_

/*
 * OpenCL Event object for holding the kernel execution information
 * (if executed or not, profiling information, etc.)
 * Since event objects are created in runtime, the goal of this
 * implementation is to minimize any overhead and do the minimum
 * as a wrapper. 
 *
 * Author: Radford Juang 
 * Date:  5.8.2010
 * Email: rayver /_at_/ hkn (dot) berkeley (dot) edu 
 */

#include <CL/cl.h>
#include <ray/opencl/OCLUtils.h>

#include <vector>

namespace ray { namespace opencl {
typedef struct _OCLEventProfile {
	cl_ulong	time_queued;
	cl_ulong	time_submit;
	cl_ulong	time_start;
	cl_ulong	time_end;
} OCLEventProfile;

class OCLEvent : public OCLObject<cl_event> {	
	/** Event passed in must be released manually. 
		However, the event is retained during lifetime of OCLEvent object */
public:
	OCLEvent() {}
	OCLEvent(cl_event evt) : OCLObject<cl_event>(evt) { }

public:
	inline cl_ulong			get_time_queued()   { 
		cl_ulong t;
		ocl_check_fast(
			clGetEventProfilingInfo(m_id, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &t, NULL),
			"clGetEventProfilingInfo - CL_PROFILING_COMMAND_QUEUED"
		);
		return t;
	}

	inline cl_ulong			get_time_submit()   { 
		cl_ulong t;
		ocl_check_fast(
			clGetEventProfilingInfo(m_id, CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong), &t, NULL),
			"clGetEventProfilingInfo - CL_PROFILING_COMMAND_SUBMIT"
		);
		return t;	
	}

	inline cl_ulong			get_time_start()    { 
		cl_ulong t;
		ocl_check_fast(
			clGetEventProfilingInfo(m_id, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &t, NULL),
			"clGetEventProfilingInfo - CL_PROFILING_COMMAND_START"
		);
		return t;	
	}

	inline cl_ulong			get_time_end()      { 
		cl_ulong t;
		ocl_check_fast(
			clGetEventProfilingInfo(m_id, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &t, NULL),
			"clGetEventProfilingInfo - CL_PROFILING_COMMAND_END"
		);
		return t;	
	}

	inline OCLEventProfile	get_times()			{ 
		OCLEventProfile prof;
		prof.time_queued = get_time_queued();
		prof.time_submit = get_time_submit();
		prof.time_start = get_time_start();
		prof.time_end = get_time_end();
		return prof;
	}

	inline cl_command_queue get_cmd_queue()   { 
		cl_command_queue v;
		ocl_check_fast( 
			clGetEventInfo(m_id, CL_EVENT_COMMAND_QUEUE, sizeof(v), &v, NULL),
			"clGetEventInfo -- CL_EVENT_COMMAND_QUEUE"		
		);
		return v;
	}

	inline cl_command_type  get_cmd_type()    { 
		cl_command_type v;
		ocl_check_fast( 
			clGetEventInfo(m_id, CL_EVENT_COMMAND_TYPE, sizeof(v), &v, NULL),
			"clGetEventInfo -- CL_EVENT_COMMAND_TYPE"		
		);
		return v;
	}

	inline cl_int			get_exec_status() { 
		cl_int v;
		ocl_check_fast( 
			clGetEventInfo(m_id, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(v), &v, NULL),
			"clGetEventInfo -- CL_EVENT_COMMAND_EXECUTION_STATUS"		
		);
		return v;
	}

	inline cl_uint			get_refcount()    { 
		cl_uint v;
		ocl_check_fast( 
			clGetEventInfo(m_id, CL_EVENT_REFERENCE_COUNT, sizeof(v), &v, NULL),
			"clGetEventInfo -- CL_EVENT_REFERENCE_COUNT"		
		);
		return v;	
	}

	inline void wait( ) { 
		waitFor(&m_id, 1);
	}

	inline static void waitFor(std::vector<OCLEvent> &events) {
		std::vector<cl_event> evt;
		evt.resize(events.size());
		for (size_t i=0; i<events.size(); ++i) {
			evt[i] = events[i].id();
		}
		waitFor(evt);
	}

	inline static void waitFor(std::vector<cl_event> &events) {
		waitFor(&events[0], events.size());
	}

	inline static void waitFor(cl_event *events, int num_events) {
		ocl_check_fast(
			clWaitForEvents(num_events, events),
			"clWaitForEvents"
		);
	}

};


}}
#endif
