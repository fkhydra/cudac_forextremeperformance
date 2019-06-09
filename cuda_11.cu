#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void set_value(int *c, int number)
{
    c[0] = number;
}

int main()
{
	int *dev0_data;
	int *dev1_data;
	int data=0;

	cudaSetDevice(0);
	cudaMalloc((void**)&dev0_data, 1 * sizeof(int));
	cudaMemcpy(dev0_data, &data, 1 * sizeof(int), cudaMemcpyHostToDevice);	
	cudaSetDevice(1);
	cudaMalloc((void**)&dev1_data, 1 * sizeof(int));	
	cudaMemcpy(dev0_data, &data, 1 * sizeof(int), cudaMemcpyHostToDevice);
	printf("Memory allocated...\n");

	cudaSetDevice(0);
	set_value << <1, 1 >> > (dev0_data,1);
	cudaDeviceSynchronize();
	cudaSetDevice(1);
	set_value << <1, 1 >> > (dev1_data, 9);
	cudaDeviceSynchronize();
	printf("Kernels ok...\n");

	cudaSetDevice(0);
	cudaMemcpy(&data, dev0_data, 1 * sizeof(int), cudaMemcpyDeviceToHost);
	printf("DEV0: %i\n", data);
	cudaSetDevice(1);
	cudaMemcpy(&data, dev1_data, 1 * sizeof(int), cudaMemcpyDeviceToHost);
	printf("DEV1: %i\n", data);

	cudaMemcpyPeer(dev0_data,0, dev1_data,1,1*sizeof(int));
	printf("Swap ok...\n");

	cudaSetDevice(0);
	cudaMemcpy(&data, dev0_data, 1 * sizeof(int), cudaMemcpyDeviceToHost);
	printf("DEV0: %i\n",data);
	cudaSetDevice(1);
	cudaMemcpy(&data, dev1_data, 1 * sizeof(int), cudaMemcpyDeviceToHost);
	printf("DEV1: %i\n", data);
	
	cudaFree(dev0_data);
	cudaFree(dev1_data);
    return 0;
}
