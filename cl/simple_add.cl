
__kernel void add(__global const float *x, __global const float *y, __global float *z, int N) {
  int id = get_global_id(0);
  if (id >= N) return;
  z[id] = x[id] + y[id];
}
