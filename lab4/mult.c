#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MEM_SIZE (5000)
#define MAX_SOURCE_SIZE (0x100000)

void gen_element(float *element)
{
	int ran1, ran2;
	size_t i, j;
	srand(clock());
	ran1 = rand() / 1000;
	for (i=0;i<MEM_SIZE;i++)
	{
		for(j=0;j<MEM_SIZE;j++)
		{
			ran2 = rand();
			element[i*MEM_SIZE + j] = (float)ran1/(float)ran2;
		} }
}

int main(void)
{
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_context context = NULL;
    cl_command_queue command_queue = NULL;
    cl_mem memobj = NULL, memobj2 = NULL, memobj3 = NULL;
    cl_program program = NULL;
    cl_kernel kernel = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret;

    //float mem[MEM_SIZE*MEM_SIZE], mem2[MEM_SIZE*MEM_SIZE], mem3[MEM_SIZE], mem4[MEM_SIZE];
    float *mem, *mem2, *mem3, *mem4;

    mem = (float*)malloc(sizeof(float) * MEM_SIZE * MEM_SIZE);
    mem2 = (float*)malloc(sizeof(float) * MEM_SIZE * MEM_SIZE);
    mem3 = (float*)malloc(sizeof(float) * MEM_SIZE);
    mem4 = (float*)malloc(sizeof(float) * MEM_SIZE);

    FILE *fp;
    const char fileName[] = "./kernel.cl";
    size_t source_size;
    char *source_str;
    cl_int i;

    fp = fopen(fileName, "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char *)malloc(MAX_SOURCE_SIZE);
    source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp );
    fclose( fp );

    gen_element(mem);
    gen_element(mem2);

    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);

    context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);

    command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

    memobj = clCreateBuffer(context, CL_MEM_READ_WRITE, MEM_SIZE * MEM_SIZE * sizeof(float), NULL, &ret);
    memobj2 = clCreateBuffer(context, CL_MEM_READ_WRITE, MEM_SIZE * MEM_SIZE * sizeof(float), NULL, &ret);
    memobj3 = clCreateBuffer(context, CL_MEM_READ_WRITE, MEM_SIZE * sizeof(float), NULL, &ret);

    ret = clEnqueueWriteBuffer(command_queue, memobj, CL_TRUE, 0, MEM_SIZE * MEM_SIZE * sizeof(float), mem, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, memobj2, CL_TRUE, 0, MEM_SIZE * MEM_SIZE * sizeof(float), mem2, 0, NULL, NULL);

    program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);

    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (ret != CL_SUCCESS)
    {
	    printf("Kernel build error %d\n", ret );
    }

    kernel = clCreateKernel(program, "vecAdd", &ret);

    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&memobj);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&memobj2);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&memobj3);
    int size = MEM_SIZE;
    ret = clSetKernelArg(kernel, 3, sizeof(int), (void *)&size);

    size_t global_work_size[3] = {MEM_SIZE, 0, 0};
    size_t local_work_size[3]  = {MEM_SIZE, 0, 0};

    clock_t time = clock();
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);
    time = clock() - time;

    ret = clEnqueueReadBuffer(command_queue, memobj3, CL_TRUE, 0, MEM_SIZE * sizeof(float), mem3, 0, NULL, NULL);


    printf("Kernel time %d\n", time);

    time = clock();

    for (i=0; i<MEM_SIZE; i++) 
    {
	   mem4[i] = mem[i] * mem2[i*MEM_SIZE];
    }

    time = clock() - time;

    printf("C time %d\n", time);

    for(i=0; i<MEM_SIZE; i++) {
	    if (mem[i] * mem2[i*MEM_SIZE] != mem3[i])
        	printf("%d : %f * %f == %f\n", i, mem[i], mem2[i*MEM_SIZE], mem3[i]);
    }

    printf("Succecc\n");
    free(mem);
    free(mem2);
    free(mem3);
    free(mem4);

    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(memobj);
    ret = clReleaseMemObject(memobj2);
    ret = clReleaseMemObject(memobj3);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);

    free(source_str);

    return 0;
}

