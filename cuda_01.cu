#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void GPU_kernel(int maxcount)
{
	int i;
	int startindex = threadIdx.x + (blockIdx.x * blockDim.x);
	int step = blockDim.x * gridDim.x;
	for (i = startindex; i < maxcount; i += step)
	{
		printf("%i\n", i);
	}
}

int main(void)
{
	int threads = 128;
	int blocks = (1000000 + threads - 1) / threads;

	GPU_kernel << < blocks , threads >> > (1000000);
	cudaDeviceSynchronize();
	printf("Execution done!\n");
	return 0;
}
