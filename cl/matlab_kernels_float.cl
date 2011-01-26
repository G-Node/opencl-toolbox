
inline int get_index(int nelems, int index ) {
/* Fetch a linear index to work on given # of elements
 * and previous index. -1 if start
 */

  if (index == -1)  {
    index = get_global_id(0);
  } else {
    index += get_global_size(0);
  }

  if (index >= nelems) index = -1;

  return index;
}

__kernel void single_add(__global float *out, __global const float *x, __global const float *y, int N) {
  int id = get_index(N, -1);
  while(id >= 0) {
    out[id] = x[id] + y[id];
    id = get_index(N, id);
  }
}

__kernel void single_minus(__global float *out, __global const float *x, __global const float *y, int N) {
  int id = get_index(N, -1);
  while(id >= 0) {
    out[id] = x[id] - y[id];
    id = get_index(N, id);
  }
}

__kernel void single_divide(__global float *out, __global const float *x, __global const float *y, int N) {
  int id = get_index(N, -1);
  while(id >= 0) {
    out[id] = x[id] /  y[id];
    id = get_index(N, id);
  }
}

__kernel void single_times(__global float *out, __global const float *x, __global const float *y, int N) {
  int id = get_index(N, -1);
  while(id >= 0) {
    out[id] = x[id] *  y[id];
    id = get_index(N, id);
  }
}

__kernel void single_scalar_times(__global float *out, float w, global const float *x, int N) {
  int id = get_index(N, -1);
  while(id >= 0) {
    out[id] = w * x[id]; 
    id = get_index(N, id);
  }
}

__kernel void single_times_scalar(__global float *out, global const float *x, float w, int N) {
  int id = get_index(N, -1);
  while(id >= 0) {
    out[id] = w * x[id]; 
    id = get_index(N, id);
  }
}

__kernel void single_add_scalar(__global float *out, global const float *x, float w, int N) {
  int id = get_index(N, -1);
  while(id >= 0) {
    out[id] = w + x[id]; 
    id = get_index(N, id);
  }
}

__kernel void single_scalar_add(__global float *out, float w, global const float *x, int N) {
  int id = get_index(N, -1);
  while(id >= 0) {
    out[id] = w + x[id]; 
    id = get_index(N, id);
  }
}

__kernel void single_scalar_minus(__global float *out, float w, global const float *x, int N) {
  int id = get_index(N, -1);
  while(id >= 0) {
    out[id] = w - x[id]; 
    id = get_index(N, id);
  }
}

__kernel void single_minus_scalar(__global float *out, global const float *x, float w, int N) {
  int id = get_index(N, -1);
  while(id >= 0) {
    out[id] = x[id] - w; 
    id = get_index(N, id);
  }
}

__kernel void single_scalar_divide(__global float *out, float w, global const float *x, int N) {
  int id = get_index(N, -1);
  while(id >= 0) {
    out[id] = w / x[id]; 
    id = get_index(N, id);
  }
}

__kernel void single_divide_scalar(__global float *out, global const float *x, float w, int N) {
  int id = get_index(N, -1);
  while(id >= 0) {
    out[id] = x[id] / w; 
    id = get_index(N, id);
  }
}

__kernel void single_exponential(__global float *out, __global const float *x, int N) {
  int id = get_index(N, -1);
  while(id >= 0) {
    out[id] = exp(x[id]);
    id = get_index(N, id);
  }
}


