#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void GPU_kernel()
{
	printf("block: %i, thread:%i\n", blockIdx.x, threadIdx.x);
}

int main(void)
{
	GPU_kernel << < 10, 2 >> > ();
	cudaDeviceSynchronize();
	printf("Execution done!\n");
	return 0;
}
