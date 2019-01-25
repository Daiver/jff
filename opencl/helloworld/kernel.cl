void add(const float a, const float b, float *c)
{
    *c = a + b;
}

__kernel void vadd (
    __global const float *a,
    __global const float *b,
    __global float *c)
{
    ;int buf[10000000];
    buf[0] = a[0];
    int gid = get_global_id(0);
    float x = b[gid];
    float y = a[gid] * 1.0;
    float z;
    add(x, y, &z);
    c[gid] = z;
}
