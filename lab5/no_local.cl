// matrixTranspose_kernel.cl
// Kernel source file for calculating the transpose of a matrix

__kernel
void matrix_sum(__global float * output,
                     const __global float * inputA,
                     const __global float * inputB,
		     __local float * block,
                     const uint width,
		     const uint blockSize)

{

	uint globalIdx = get_global_id(0);
	uint globalIdy = get_global_id(1);

	uint targetIndex = globalIdy * width + globalIdx;
	uint sourceIndex = globalIdx * width + globalIdy;

	output[targetIndex] = inputA[sourceIndex] + inputB[targetIndex];

}

