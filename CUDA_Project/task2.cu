#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include<math.h>
#include <time.h>
#include <windows.h>
#include <iostream>
#include <chrono>
using namespace std;
using namespace chrono;

cudaError_t addWithCuda(const float *index, float *delta, float *component, unsigned int size);
void cuda_cpu();
double calGaussianFunction(double x);

__global__ void initDelta(const float *index, float *res)
{
	int i = threadIdx.x;
	res[i] = 10.0 / index[i];
}


int main() {
	cuda_cpu();
	/*const int arraySize = 10;
	const float index[arraySize] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 };
	float delta[arraySize] = { 0 };
	float component[arraySize] = { 0 };
	cudaError_t cudaStatus = addWithCuda(index, delta, component, arraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}
	printf("{%d,%d,%d,%d,%d,%d,%d,%d,%d,%d}\n", delta[0], delta[1], delta[2], delta[3], delta[4], delta[5], delta[6], delta[7], delta[8], delta[9]);*/
}


void cuda_cpu()
{
	double cost;
	const int arraySize = 6;
	const double a[arraySize] = { -3, -2, -1, 0, 1, 2};
	double result = 0;
	double temp = 0;
	double x = 0;
	printf("----------------CPU code----------------\r\n");
	for (int i = 0; i < arraySize; i++)
	{
		auto start = system_clock::now();
		x = a[i] + 0.5;
		temp = calGaussianFunction(x);
		printf("Step %d - temp:%f\r\n", i, temp);
		auto end = system_clock::now();
		result += temp;
		printf("Step %d - sum:%f\r\n",i, result);
		auto duration = duration_cast<microseconds>(end - start);
		double t = double(duration.count()) * microseconds::period::num / microseconds::period::den;
		printf("Step %d - time: %f\r\n\r\n", i, t);
	}
}

double calGaussianFunction(double x)
{
	double pow_result = pow(x, 2.0) * -1;
	return exp(pow_result);
}

//main algorithm
cudaError_t addWithCuda(const float *index, float *delta, float *component, unsigned int size)
{
	float *dev_index = 0;
	float *dev_delta = 0;
	float *dev_component = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_index, size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_delta, size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_component, size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_index, index, size * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	initDelta <<<1, size>>> (dev_index, dev_delta);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(delta, dev_delta, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_index);
	cudaFree(dev_delta);
	cudaFree(dev_component);

	return cudaStatus;
}
