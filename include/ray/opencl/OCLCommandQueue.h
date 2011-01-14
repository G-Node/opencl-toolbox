#ifndef _RAY_OPENCL_OCLCOMMANDQUEUE_H_
#define _RAY_OPENCL_OCLCOMMANDQUEUE_H_

/*
 * OpenCL CommandQueue object for managing commands 
 * 
 * A command queue is responsible for sending commands to a device.
 * Each device can be associated with multiple context, so the command queue
 * is specific to a context. 
 *  
 * Author: Radford Juang 
 * Date:  5.7.2010
 * Email: rayver /_at_/ hkn (dot) berkeley (dot) edu
 */

#include <ray/opencl/OCLConstants.h>
#include <ray/opencl/OCLContext.h>
#include <ray/opencl/OCLDevice.h>
#include <ray/opencl/OCLError.h>
#include <ray/opencl/OCLUtils.h>
#include <ray/opencl/OCLKernel.h>


namespace ray { namespace opencl {

class OCLCommandQueue : public OCLObject<cl_command_queue> {
public:
	cl_device_id				m_device;
	cl_context					m_context;
	cl_command_queue_properties m_properties;
	cl_uint						m_refcount;
public:

	OCLCommandQueue(cl_command_queue id) : OCLObject<cl_command_queue>(id) { 
		query_info(); 
	}

	OCLCommandQueue(cl_context context, cl_device_id device, cl_command_queue_properties properties = 0) : 
		m_device(device), m_context(context), m_properties(properties){
		create();
	}

	OCLCommandQueue(OCLContext &context, OCLDevice &device, cl_command_queue_properties properties = 0) : 
		m_device(device.id()), m_context(context.id()), m_properties(properties){
		create();
	}

	OCLCommandQueue(OCLContext *context, OCLDevice *device, cl_command_queue_properties properties = 0) : 
		m_device(device->id()), m_context(context->id()), m_properties(properties){
		create();
	}

	OCLCommandQueue(OCLContext &context, cl_device_id device, cl_command_queue_properties properties = 0) : 
		m_device(device), m_context(context.id()), m_properties(properties){
		create();
	}

	OCLCommandQueue(OCLContext *context, cl_device_id device, cl_command_queue_properties properties = 0) : 
		m_device(device), m_context(context->id()), m_properties(properties){
		create();
	}


	inline void set(cl_device_id device )	{ m_device = device; }
	inline void set(cl_context context)		{ m_context = context; }

	inline void set(OCLDevice &device )		{ m_device = device.id(); }
	inline void set(OCLContext &context)	{ m_context = context.id(); }	

	inline void set(OCLDevice *device )		{ m_device = device->id(); }
	inline void set(OCLContext *context)	{ m_context = context->id(); }	

	inline void set(cl_command_queue_properties properties, cl_bool enable) {
		if (enable) 
			m_properties |= properties;
		else
			m_properties &= ~properties;

		set(m_properties);
	}

    //Deprecated in 1.1 API --
	inline void set(cl_command_queue_properties properties) {
		m_properties = properties;

		if (m_id) {
			//Set the properties on device if queue is already created
			cl_bool result = ((m_properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) != 0);

            // These properties are deprecated and can no longer be set in
            // 1.1+
            /*         
			ocl_check( 
				clSetCommandQueueProperty(m_id, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, result,  NULL),
				"clSetCommandQueue - CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE"
			);
           
			result = ((m_properties & CL_QUEUE_PROFILING_ENABLE) != 0);
            
			ocl_check( 
				clSetCommandQueueProperty(m_id, CL_QUEUE_PROFILING_ENABLE, result,  NULL),
				"clSetCommandQueue - CL_QUEUE_PROFILING_ENABLE"
			);
            */
		}
	}
    

	inline void create() {
		if (m_id) release();

		cl_int errcode = CL_SUCCESS;
		m_id = clCreateCommandQueue(m_context, m_device, m_properties, &errcode);
		ocl_check(errcode, "clCreateCommandQueue");
		query_info();
	}


	//The following functions have to be executed in real time and as such will have to have a fast
	// implementation

	inline void enqueue_buffer_copy(void *dst, cl_mem src, size_t num_bytes, 
				size_t  buff_byte_offset	 = 0,
			   cl_bool	blocking			 = CL_FALSE, 
			   cl_uint  num_events_to_wait   = 0,
		const cl_event *event_waitlist		 = NULL,
			  OCLEvent *event_out			 = NULL
		) {
		cl_event e;

		if (event_out) {
			ocl_check_fast(
					clEnqueueReadBuffer(m_id, src, blocking, buff_byte_offset, num_bytes, dst,
										num_events_to_wait, event_waitlist, &e),
					"clEnqueueReadBuffer"
				);
			event_out->assign(e);
		} else {
			ocl_check_fast(
					clEnqueueReadBuffer(m_id, src, blocking, buff_byte_offset, num_bytes, dst,
										num_events_to_wait, event_waitlist, NULL),
					"clEnqueueReadBuffer"
				);
		}
	}

	inline void enqueue_buffer_copy(cl_mem dst, const void *src, size_t num_bytes, 
				size_t  buff_byte_offset	 = 0,
			   cl_bool	blocking			 = CL_FALSE, 
			   cl_uint  num_events_to_wait   = 0,
		const cl_event *event_waitlist		 = NULL,
			  OCLEvent *event_out			 = NULL
		) {
		cl_event e;
		if (event_out) {
			ocl_check_fast(
				clEnqueueWriteBuffer(m_id, dst, blocking, buff_byte_offset, num_bytes, src,
									num_events_to_wait, event_waitlist, &e),
				"clEnqueueWriteBuffer"
			);
			event_out->assign(e);
		} else {
			ocl_check_fast(
				clEnqueueWriteBuffer(m_id, dst, blocking, buff_byte_offset, num_bytes, src,
									num_events_to_wait, event_waitlist, NULL),
				"clEnqueueWriteBuffer"
			);
		}
	}


	inline void enqueue_buffer_copy(cl_mem dst, cl_mem src, size_t num_bytes, 
				size_t  dst_byte_offset	 = 0,
				size_t  src_byte_offset	 = 0,			   
			   cl_uint  num_events_to_wait   = 0,
		const cl_event *event_waitlist		 = NULL,
			  OCLEvent *event_out			 = NULL
		) {
		cl_event e;
		if (event_out) {
			ocl_check_fast(
				clEnqueueCopyBuffer(m_id, src, dst, src_byte_offset, dst_byte_offset, num_bytes, 
									num_events_to_wait, event_waitlist, &e),
				"clEnqueueCopyBuffer"
			);
			event_out->assign(e);
		} else {
			ocl_check_fast(
				clEnqueueCopyBuffer(m_id, src, dst, src_byte_offset, dst_byte_offset, num_bytes, 
									num_events_to_wait, event_waitlist, NULL),
				"clEnqueueCopyBuffer"
			);
		}
	}

	inline void enqueue_buffer_copy(void *dst, OCLBuffer &src, size_t num_bytes, 
				size_t  buff_byte_offset	 = 0,
			   cl_bool	blocking			 = CL_FALSE, 
			   cl_uint  num_events_to_wait   = 0,
		const cl_event *event_waitlist		 = NULL,
			  OCLEvent *event_out			 = NULL
	) { 	
		enqueue_buffer_copy(dst,src.id(),num_bytes, buff_byte_offset, blocking, num_events_to_wait, event_waitlist, event_out);
	}

	inline void enqueue_buffer_copy(OCLBuffer &dst, const void *src, size_t num_bytes, 
				size_t  buff_byte_offset	 = 0,
			   cl_bool	blocking			 = CL_FALSE, 
			   cl_uint  num_events_to_wait   = 0,
		const cl_event *event_waitlist		 = NULL,
			  OCLEvent *event_out			 = NULL
	) {
		enqueue_buffer_copy(dst.id(), src, num_bytes, buff_byte_offset, blocking, num_events_to_wait, event_waitlist, event_out);	
	}

	inline void enqueue_buffer_copy(OCLBuffer &dst, OCLBuffer &src, size_t num_bytes, 
				size_t  dst_byte_offset		 = 0,
				size_t  src_byte_offset		 = 0,			   
			   cl_uint  num_events_to_wait   = 0,
		const cl_event *event_waitlist		 = NULL,
			  OCLEvent *event_out			 = NULL
		) {
			enqueue_buffer_copy(dst.id(), src.id(), num_bytes, dst_byte_offset, src_byte_offset, num_events_to_wait, event_waitlist, event_out);		
	}


	inline void enqueue_marker(cl_event *out_event) {
		ocl_check_fast(
			clEnqueueMarker(m_id, out_event),
			"clEnqueueMarker"
		);
	}

	inline void enqueue_marker(OCLEvent &out_event) { 
		cl_event e;

		enqueue_marker(&e);
		out_event.assign(e);

	}
	inline void enqueue_marker(OCLEvent *out_event) { 
		cl_event e;
		enqueue_marker(&e);
		out_event->assign(e);
	}

	inline void enqueue_barrier() { 
		ocl_check_fast(
			clEnqueueBarrier(m_id),
			"clEnqueueBarrier"
		);
	}

	inline void enqueue_ndrange_kernel(cl_kernel kernel, cl_uint work_dim, 
			const size_t *global_work_offset, 
			const size_t *global_work_size, 
			const size_t *local_work_size, 
			cl_uint num_events_in_waitlist = 0,
			const cl_event *event_waitlist = NULL, 
			cl_event *out_event = NULL) 
	{
		ocl_check_fast(
			clEnqueueNDRangeKernel(m_id, kernel, work_dim, global_work_offset, global_work_size, local_work_size, num_events_in_waitlist, event_waitlist, out_event),
			"clEnqueueNDRangeKernel"
		);
	}

	inline void enqueue_ndrange_kernel(cl_kernel kernel, cl_uint work_dim, 
			const size_t *global_work_offset, 
			const size_t *global_work_size, 
			const size_t *local_work_size, 
			cl_uint num_events_in_waitlist = 0,
			const cl_event *event_waitlist = NULL, 
			OCLEvent *out_event = NULL) 
	{
		cl_event e;
		if (out_event) {
			ocl_check_fast(
				clEnqueueNDRangeKernel(m_id, kernel, work_dim, global_work_offset, global_work_size, local_work_size, num_events_in_waitlist, event_waitlist, &e),
				"clEnqueueNDRangeKernel"
			);
			out_event->assign(e);
		} else {
			ocl_check_fast(
				clEnqueueNDRangeKernel(m_id, kernel, work_dim, global_work_offset, global_work_size, local_work_size, num_events_in_waitlist, event_waitlist, NULL),
				"clEnqueueNDRangeKernel"
			);
		}
	}

	inline void enqueue_ndrange_kernel(OCLKernel &kernel, OCLEvent *out_event=NULL) {
		cl_event e;
		if (out_event) {
			ocl_check_fast(
				clEnqueueNDRangeKernel(m_id, kernel.id() , kernel.m_num_dims, NULL, kernel.m_global_group_size, kernel.m_local_group_size, 0, NULL, &e),
				"clEnqueueNDRangeKernel"
			);		
			out_event->assign(e);
		} else {
			ocl_check_fast(
				clEnqueueNDRangeKernel(m_id, kernel.id() , kernel.m_num_dims, NULL, kernel.m_global_group_size, kernel.m_local_group_size, 0, NULL, NULL),
				"clEnqueueNDRangeKernel"
			);		
		}
	}

	inline void enqueue_ndrange_kernel(OCLKernel *kernel, OCLEvent *out_event=NULL) {
		cl_event e;
		if (out_event) {
			ocl_check_fast(
				clEnqueueNDRangeKernel(m_id, kernel->id() , kernel->m_num_dims, NULL, kernel->m_global_group_size, kernel->m_local_group_size, 0, NULL, &e),
				"clEnqueueNDRangeKernel"
			);		
			out_event->assign(e);
		} else {
			ocl_check_fast(
				clEnqueueNDRangeKernel(m_id, kernel->id() , kernel->m_num_dims, NULL, kernel->m_global_group_size, kernel->m_local_group_size, 0, NULL, NULL),
				"clEnqueueNDRangeKernel"
			);		
		}
	}


	inline void enqueue_ndrange_kernel(OCLKernel &kernel, std::vector<cl_event> &events_to_wait_on, OCLEvent *out_event=NULL) {
		cl_event e;
		if (out_event) {
			ocl_check_fast(
				clEnqueueNDRangeKernel(m_id, kernel.id() , kernel.m_num_dims, NULL, kernel.m_global_group_size, kernel.m_local_group_size, events_to_wait_on.size() , &events_to_wait_on[0], &e),
				"clEnqueueNDRangeKernel"
			);		
			out_event->assign(e);
		} else {
			ocl_check_fast(
				clEnqueueNDRangeKernel(m_id, kernel.id() , kernel.m_num_dims, NULL, kernel.m_global_group_size, kernel.m_local_group_size, events_to_wait_on.size() , &events_to_wait_on[0] , NULL),
				"clEnqueueNDRangeKernel"
			);		
		}
	}

	inline void enqueue_ndrange_kernel(OCLKernel *kernel, std::vector<cl_event> &events_to_wait_on, OCLEvent *out_event=NULL) {
		cl_event e;
		if (out_event) {
			ocl_check_fast(
				clEnqueueNDRangeKernel(m_id, kernel->id() , kernel->m_num_dims, NULL, kernel->m_global_group_size, kernel->m_local_group_size, events_to_wait_on.size() , &events_to_wait_on[0], &e),
				"clEnqueueNDRangeKernel"
			);		
			out_event->assign(e);
		} else {
			ocl_check_fast(
				clEnqueueNDRangeKernel(m_id, kernel->id() , kernel->m_num_dims, NULL, kernel->m_global_group_size, kernel->m_local_group_size, events_to_wait_on.size() , &events_to_wait_on[0], NULL),
				"clEnqueueNDRangeKernel"
			);		
		}
	}
/*
	inline void enqueue_task(cl_kernel kernel) {
		//TODO
	}

	inline void enqueue_native_kernel() {
		//TODO
	}
*/

	inline void enqueue_waitfor_events(std::vector<cl_event> &events) {
		ocl_check_fast(
			clEnqueueWaitForEvents(m_id, events.size(), &events[0]),
			"clEnqueueWaitForEvents"
		);
	}

	inline void enqueue_waitfor_events(cl_uint num_events, const cl_event *events) {
		ocl_check_fast(
			clEnqueueWaitForEvents(m_id, num_events, events),
			"clEnqueueWaitForEvents"
		);
	}

	inline void flush() {
		ocl_check_fast(
			clFlush(m_id),
			"clFlush"
		);
	}

	inline void finish() {
		ocl_check_fast(
			clFinish(m_id),
			"clFinish"
		);
	}
	
protected:

	inline void query_info() {
		ocl_get_info(m_id, CL_QUEUE_CONTEXT, m_context, cl_context,   clGetCommandQueueInfo);
		ocl_get_info(m_id, CL_QUEUE_DEVICE,  m_device,  cl_device_id, clGetCommandQueueInfo);
		ocl_get_info(m_id, CL_QUEUE_REFERENCE_COUNT, m_refcount, cl_uint, clGetCommandQueueInfo);
		ocl_get_info(m_id, CL_QUEUE_PROPERTIES, m_properties, cl_command_queue_properties, clGetCommandQueueInfo);
	}

};

}}
#endif
