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

cudaError_t caclWithCuda(const float *index, float *res, unsigned int size);
void cpu_code();
void gpu_code();
void getGPUInfo();
float calGaussianFunction(float x);

__global__ void initDelta(const float *index, float *res)
{
	int i = threadIdx.x;
	res[i] = (float)exp(pow((double)(index[i] + 0.5), 2.0) * -1);
}


int main() {
	getGPUInfo();
	cpu_code();
	gpu_code();
}

void getGPUInfo() {
	printf("----------------GPU Info----------------\r\n");
	cudaError_t cudaStatus;
	int num;
	cudaDeviceProp prop;
	cudaStatus = cudaGetDeviceCount(&num);
	printf("deviceCount := %d\n", num);
	for (int i = 0; i < num; i++) {
		cudaGetDeviceProperties(&prop, i);
		printf("name:%s\n", prop.name);
		printf("totalGlobalMem:%d byte\n", prop.totalGlobalMem);
		printf("multiProcessorCount:%d\n", prop.multiProcessorCount);
		printf("maxThreadsPerBlock:%d\n", prop.maxThreadsPerBlock);
	}
}

void cpu_code()
{
	const int arraySize = 6;
	float a[arraySize] = { -3, -2, -1, 0, 1, 2 };
	float sum = 0;
	float temp = 0;
	float x = 0;
	auto init_time = system_clock::now();
	printf("----------------CPU code----------------\r\n");
	for (int i = 0; i < arraySize; i++)
	{
		auto start = system_clock::now();
		x = a[i] + 0.5;
		temp = calGaussianFunction(x);
		printf("Step %d - temp:%f\r\n", i, temp);
		auto end = system_clock::now();
		sum += temp;
		printf("Step %d - sum:%f\r\n", i, sum);
		auto duration = duration_cast<microseconds>(end - start);
		float t = float(duration.count()) * microseconds::period::num / microseconds::period::den;
		printf("Step %d - time: %f\r\n\r\n", i, t);
	}
	auto final_time = system_clock::now();
	auto duration = duration_cast<microseconds>(final_time - init_time);
	float t = float(duration.count()) * microseconds::period::num / microseconds::period::den;
	printf("# Total time: %f\r\n\r\n", t);
	printf("# Area:%f\r\n\r\n", sum);
}

void gpu_code() {
	printf("----------------GPU code----------------\r\n");
	const int arraySize = 6;
	float x[arraySize] = { -3, -2, -1, 0, 1, 2 };
	float res[arraySize] = { 0 };
	float sum = 0;
	cudaError_t cudaStatus = caclWithCuda(x, res, arraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return;
	}
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return;
	}
	for (int i = 0; i < arraySize; i++) {
		sum += res[i];
	}
	printf("# Area:%f\r\n", sum);
}

float calGaussianFunction(float x)
{
	float pow_result = pow(x, 2.0) * -1;
	return exp(pow_result);
}

//main algorithm
cudaError_t caclWithCuda(const float *index, float *res, unsigned int size)
{
	float *dev_index = 0;
	float *dev_res = 0;
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

	cudaStatus = cudaMalloc((void**)&dev_res, size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_index, index, size * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	auto start = system_clock::now();
	// <<<blocks, threads>>>
	// initDelta <<<size, 1>>> (dev_index, dev_res);
	// initDelta <<<2, 3>>> (dev_index, dev_res);
	// initDelta <<<3, 2>>> (dev_index, dev_res);
	initDelta <<<1, size>>> (dev_index, dev_res);
	auto end = system_clock::now();
	auto duration = duration_cast<microseconds>(end - start);
	float t = float(duration.count()) * microseconds::period::num / microseconds::period::den;
	printf("# Total time: %f\r\n\r\n", t);

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
	cudaStatus = cudaMemcpy(res, dev_res, size * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_index);
	cudaFree(dev_res);

	return cudaStatus;
}
