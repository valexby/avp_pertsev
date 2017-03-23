__kernel void vecAdd(__global float* a,  __global float* b, __global float* c, int size)
{
	int gid = get_global_id(0);
	if (gid > size) return;

	c[gid] = a[gid] * b[gid*size];
	//a[gid] += b[gid];
	//a[gid] =b[gid] ;

}
