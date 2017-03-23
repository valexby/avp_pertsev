// matrixTranspose_kernel.cl
// Kernel source file for calculating the transpose of a matrix

__kernel
void matrix_sum(__global float * output,
                     const __global float * inputA,
                     const __global float * inputB,
		     __local float * block,
                     const uint size,
		     const uint block_size)

{

	uint global_id_x = get_global_id(0);
	uint global_id_y = get_global_id(1);

	uint local_id_x = get_local_id(0);
	uint local_id_y = get_local_id(1);

	block [local_id_y * block_size + local_id_x] = inputA [global_id_y * size + global_id_x];

	barrier (CLK_LOCAL_MEM_FENCE);

	uint group_id_x = get_group_id(0);
	uint group_id_y = get_group_id(1);

	uint target_global_id_x = group_id_y * block_size + local_id_y;
	uint target_global_id_y = group_id_x * block_size + local_id_x;

	uint target_index = target_global_id_y * size + target_global_id_x;
	uint source_index = local_id_y * block_size + local_id_x;

	output[target_index] = block[source_index] + inputB[target_index];

}
