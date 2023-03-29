#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void less_equal(__global half *in, __global half *comp, __global half *out) {
    int id = get_global_id(0);
    out[id] = (half) (in[id] <= comp[id]);
}

__kernel void less(__global half *in, __global half *comp, __global half *out) {
    int id = get_global_id(0);
    out[id] = (half) (in[id] < comp[id]);
}

__kernel void greater_equal(__global half *in, __global half *comp, __global half *out) {
    int id = get_global_id(0);
    out[id] = (half) (in[id] >= comp[id]);
}

__kernel void greater(__global half *in, __global half *comp, __global half *out) {
    int id = get_global_id(0);
    out[id] = (half) (in[id] > comp[id]);
}

__kernel void equal(__global half *in, __global half *comp, __global half *out) {
    int id = get_global_id(0);
    out[id] = (half) (in[id] == comp[id]);
}

__kernel void not_equal(__global half *in, __global half *comp, __global half *out) {
    int id = get_global_id(0);
    out[id] = (half) (in[id] != comp[id]);
}