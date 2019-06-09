#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#define SCREEN_WIDTH 6
#define SCREEN_HEIGHT 4

__global__ void GPU_kernel(int maxcount)
{
	int x = threadIdx.x + (blockIdx.x * blockDim.x);
	int y = threadIdx.y + (blockIdx.y * blockDim.y);
	int aktualis_index = x + (y * blockDim.x * gridDim.x);
	
	if (aktualis_index >= maxcount) return;
	printf("Actual pixel: %i , %i\n", x, y);
}

int main(void)
{
	int ThreadsX = 3, ThreadsY = 2;
	dim3 blokktomb(( SCREEN_WIDTH + ThreadsX - 1) / ThreadsX, (SCREEN_HEIGHT + ThreadsY - 1) / ThreadsY);
	dim3 szaltomb(ThreadsX, ThreadsY);

	GPU_kernel <<< blokktomb, szaltomb >>> ( SCREEN_WIDTH * SCREEN_HEIGHT);
	cudaDeviceSynchronize();
	printf("Execution done!\n");
	return 0;
}
