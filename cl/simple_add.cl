// Need to define a simplified iteration id:
inline int get_index(int nelems, int index ) { 
  if (index == -1)  {
    index = get_global_id(0);
  } else {
    index += get_global_size(0);
  }

  if (index >= nelems) index = -1;

  return index;
}

__kernel void add(__global const float *x, __global const float *y, __global float *z, int N) {

  int id = get_index(N, -1);
  while(id >= 0) {
     z[id] = x[id] + y[id];
     id = get_index(N, id);
  }

}
