#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "CL/cl.h"
#include <unistd.h>

#define SIZE 512
#define BLOCK_SZ 4

static const char * CLErrString(cl_int);
void check_err(cl_int *, int);
void gen_element(cl_float *);
clock_t count_cl(cl_float*, cl_float*, cl_float*, const char[], cl_context, cl_command_queue, cl_device_id);

static const char * CLErrString(cl_int status) {
   static struct { cl_int code; const char *msg; } error_table[] = {
      { CL_SUCCESS, "success" },
      { CL_DEVICE_NOT_FOUND, "device not found", },
      { CL_DEVICE_NOT_AVAILABLE, "device not available", },
      { CL_COMPILER_NOT_AVAILABLE, "compiler not available", },
      { CL_MEM_OBJECT_ALLOCATION_FAILURE, "mem object allocation failure", },
      { CL_OUT_OF_RESOURCES, "out of resources", },
      { CL_OUT_OF_HOST_MEMORY, "out of host memory", },
      { CL_PROFILING_INFO_NOT_AVAILABLE, "profiling not available", },
      { CL_MEM_COPY_OVERLAP, "memcopy overlaps", },
      { CL_IMAGE_FORMAT_MISMATCH, "image format mismatch", },
      { CL_IMAGE_FORMAT_NOT_SUPPORTED, "image format not supported", },
      { CL_BUILD_PROGRAM_FAILURE, "build program failed", },
      { CL_MAP_FAILURE, "map failed", },
      { CL_INVALID_VALUE, "invalid value", },
      { CL_INVALID_DEVICE_TYPE, "invalid device type", },
      { CL_INVALID_KERNEL, "kernel is not a valid kernel object", },
      { CL_INVALID_ARG_INDEX, "arg_index is not a valid argument index", },
      { CL_INVALID_ARG_VALUE, "arg_value specified is NULL for an argument that is not declared with the __local qualifier or vice-versa", },
      { CL_INVALID_MEM_OBJECT, "for an argument declared to be a memory object when the specified arg_value is not a valid memory object.", },
      { CL_INVALID_SAMPLER, "for an argument declared to be of type sampler_t when the specified arg_value is not a valid sampler object. ", },
      { CL_INVALID_ARG_SIZE, "arg_size does not match the size of the data type for an argument that is not a memory object or if the argument is a memory object and arg_size != sizeof(cl_mem) or if arg_size is zero and the argument is declared with the __local qualifier or if the argument is a sampler and arg_size != sizeof(cl_sampler)", },
      { CL_OUT_OF_RESOURCES, "there is a failure to allocate resources required by the OpenCL implementation on the device", },
      { CL_OUT_OF_HOST_MEMORY, "there is a failure to allocate resources required by the OpenCL implementation on the host.", },
      { 0, NULL },
   };
   static char unknown[25];
   int ii;

   for (ii = 0; error_table[ii].msg != NULL; ii++) {
      if (error_table[ii].code == status) {
         return error_table[ii].msg;
      }
   }

   snprintf(unknown, sizeof unknown, "unknown error %d", status);
   return unknown;
}

void check_err(cl_int *err, int size)
{
	for (--size; size >= 0; size--)
	{
		if (err[size] != CL_SUCCESS)
		{	
			printf("Error Code=%d\n",err[size]);
			printf("%s\n", CLErrString(err[size]));
			exit(1);
		}
	}
}

void gen_element(cl_float *element)
{
	int ran1, ran2;
	size_t i, j;
	srand(clock());
	ran1 = rand() / 1000;
	for (i=0;i<SIZE;i++)
	{
		for(j=0;j<SIZE;j++)
		{
			ran2 = rand();
			element[i*SIZE + j] = (cl_float)ran1/(cl_float)ran2;
		} 
	}
}

clock_t count_cl(cl_float *result, cl_float *input_matA, cl_float *input_matB, 
		const char kern_name[], cl_context context, cl_command_queue command_queue, cl_device_id device_id)
{
	cl_uint size = SIZE;
	cl_uint bl_size = BLOCK_SZ;

	cl_int err[6];
	cl_program program;
	cl_kernel kernel;
	cl_mem input_bufferB, input_bufferA, output_buffer;
	size_t global[2];
	size_t local[2];
	
	FILE *fp;
	long filelen;
	long readlen;
	char *kernel_src;  
	
	fp = fopen(kern_name,"r");
	fseek(fp,0L, SEEK_END);
	filelen = ftell(fp);
	rewind(fp);

	kernel_src = malloc(sizeof(char)*(filelen+1));
	readlen = fread(kernel_src,1,filelen,fp);
	if(readlen!= filelen)
	{
		printf("error reading file\n");
		exit(1);
	}
	fclose(fp);

	kernel_src[filelen]='\0';
	program = clCreateProgramWithSource(context, 1 ,(const char **)
                                          &kernel_src, NULL, &(err[0]));
	check_err(err, 1);
	       
	err[0] = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

	size_t len;
	char buffer[2048];
	clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 
			sizeof(buffer), buffer, &len);
	printf("--- Build Log -- \n %s\n",buffer);

	if (err[0] != CL_SUCCESS)
	{
        	printf("Build failed. Error Code=%d\n", err[0]);
		printf("%s\n", kernel_src);
		printf("%s\n", CLErrString(err[0]));
		exit(1);
	}
	
	kernel = clCreateKernel(program, "matrix_sum", &(err[0]));
	check_err(err, 1);

	input_bufferA =  clCreateBuffer(context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                        sizeof(cl_float) * SIZE * SIZE, input_matA, NULL);
	input_bufferB =  clCreateBuffer(context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                        sizeof(cl_float) * SIZE * SIZE, input_matB, NULL);
	output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
                        sizeof(cl_float) * SIZE * SIZE, NULL ,NULL);

	err[0] = clSetKernelArg(kernel, 0, sizeof(cl_mem), &output_buffer);
	err[1] = clSetKernelArg(kernel, 1, sizeof(cl_mem), &input_bufferA);
	err[2] = clSetKernelArg(kernel, 2, sizeof(cl_mem), &input_bufferB);
	err[3] = clSetKernelArg(kernel, 3, sizeof(cl_mem), NULL);
	err[4] = clSetKernelArg(kernel, 4, sizeof(cl_uint), &size);
	err[5] = clSetKernelArg(kernel, 5, sizeof(cl_uint), &bl_size);
	check_err(err, 6);
	
	global[0]= size;
	global[1]= size;
	local[0]= bl_size;
	local[1]= bl_size;

	clock_t time = clock();
	err[0] = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, 
                                   global, local, 0, NULL, NULL);
	time = clock() - time;
	check_err(err, 1);

	clFinish(command_queue);

	err[0] = clEnqueueReadBuffer(command_queue, output_buffer, CL_TRUE, 0,
                   sizeof(cl_float)*size*size, result, 0, NULL, NULL);
	check_err(err, 1);
	
	clReleaseMemObject(input_bufferA);
	clReleaseMemObject(input_bufferB);
	clReleaseMemObject(output_buffer);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	free(kernel_src);

	return time;
}

int main(void)
{
	cl_uint size = SIZE;
	cl_float *input_matA, *input_matB;
	cl_float *results_c, *results_fast, *results_slow;

	cl_uint num_devs_returned;
	cl_device_id device_id;
	cl_int err[6];
	cl_platform_id platform_id, *platform_list;
	cl_uint num_platforms_returned;
	cl_context context;
	cl_command_queue command_queue;

	clock_t time_c, time_fast, time_slow;

	input_matA = malloc(sizeof(cl_float)*size*size);
	input_matB = malloc(sizeof(cl_float)*size*size);
	results_fast = malloc(sizeof(cl_float)*size*size);
	results_slow = malloc(sizeof(cl_float)*size*size);
	results_c = malloc(sizeof(cl_float)*size*size);

	gen_element(input_matA);
	gen_element(input_matB);

	err[0] = clGetPlatformIDs(0,NULL,&num_platforms_returned); 
	check_err(err, 1);
	platform_list = malloc(num_platforms_returned * sizeof(platform_id));
	err[0] = clGetPlatformIDs(num_platforms_returned,platform_list,NULL); 
	check_err(err, 1);
	
	num_devs_returned = 0;
	int i = -1;
	while (num_devs_returned == 0)
	{
		i++;
		err[0] = clGetDeviceIDs(platform_list[i], CL_DEVICE_TYPE_CPU, 0, 
                           NULL, &num_devs_returned);
	}
	err[0] = clGetDeviceIDs(platform_list[i], CL_DEVICE_TYPE_CPU, 1,
                           &device_id, NULL);
	check_err(err, 1);

	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &(err[0]));
	check_err(err, 1);

	command_queue = clCreateCommandQueue(context,device_id, 0, &(err[0]));
	check_err(err, 1);

	time_slow = count_cl(results_slow, input_matA, input_matB, "no_local.cl", context, command_queue, device_id);
	time_fast = count_cl(results_fast, input_matA, input_matB, "kernel.cl", context, command_queue, device_id);
	
       	time_c = clock();	
	for (int i = size - 1; i>=0; i--)
		for (int j = size - 1; j >= 0; j--)
			results_c[i * size + j] = input_matB[i * size + j] + input_matA[j * size + i];
	time_c = clock() - time_c;

	for (int i = (size * size) - 1; i>=0; i--)
		if (results_c[i] != results_fast[i] || results_slow[i] != results_fast[i])
			printf("Test failed!\n");
	
	printf("C time : %ld\nOpenCL optimized time : %ld\nOpenCL not optimized time : %ld\n", time_c, time_fast, time_slow);
	
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);
	free(input_matA);
	free(input_matB);
	free(results_fast);
	free(results_slow);
	free(results_c);
	return 0;
}

