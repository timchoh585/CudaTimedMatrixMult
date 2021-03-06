#include <stdio.h>
#include "Error.h"
#include "Matrix.h"
#include "GpuTimer.h"

#define N 512

__global__ void matrixMultiplicationKernel(Matrix<float> d_a, Matrix<float> d_b, Matrix<float> d_c)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	float tmp;
	float a, b;

	while (i<N)
	{
		j = threadIdx.y + blockDim.y * blockIdx.y;
		while (j<N)
		{
			tmp = 0.0;
			a = 0.0;
			b = 0.0;
			for (int t = 0; t < N; t++)
			{
				a = d_a.getElement(i, t);
				b = d_b.getElement(t, j);
				tmp += a * b;
			}
			d_c.setElement(i, j, tmp);
			j += blockDim.y * gridDim.y;
		}
		i += blockDim.x*gridDim.x;
	}
}

void onDevice(Matrix<float> h_a, Matrix<float> h_b, Matrix<float> h_c)
{
	Matrix<float> d_a, d_b, d_c;
	d_a.width = h_a.width;
	d_a.height = h_a.height;

	d_b.width = h_b.width;
	d_b.height = h_b.height;

	d_c.width = h_c.width;
	d_c.height = h_c.height;

	dim3 GridBlocks(8, 8);
	dim3 ThreadsBlocks(8, 8);

	GpuTimer timer;
	timer.Start();

	const int ARRAY_BYTES = N * N * sizeof(float);

	HANDLER_ERROR_ERR(cudaMalloc((void**)&d_a.elements, ARRAY_BYTES));
	HANDLER_ERROR_ERR(cudaMalloc((void**)&d_b.elements, ARRAY_BYTES));
	HANDLER_ERROR_ERR(cudaMalloc((void**)&d_c.elements, ARRAY_BYTES));

	HANDLER_ERROR_ERR(cudaMemcpy(d_a.elements, h_a.elements, ARRAY_BYTES, cudaMemcpyHostToDevice));
	HANDLER_ERROR_ERR(cudaMemcpy(d_b.elements, h_b.elements, ARRAY_BYTES, cudaMemcpyHostToDevice));

	matrixMultiplicationKernel<<<8, 64>>>(d_a, d_b, d_c);
	HANDLER_ERROR_MSG("kernel panic!!!");

	HANDLER_ERROR_ERR(cudaMemcpy(h_c.elements, d_c.elements, ARRAY_BYTES, cudaMemcpyDeviceToHost));

	timer.Stop();

	printf("Time without blocking:  %f ms\n", timer.Elapsed());

	timer.Start();

	HANDLER_ERROR_ERR(cudaMalloc((void**)&d_a.elements, ARRAY_BYTES));
	HANDLER_ERROR_ERR(cudaMalloc((void**)&d_b.elements, ARRAY_BYTES));
	HANDLER_ERROR_ERR(cudaMalloc((void**)&d_c.elements, ARRAY_BYTES));

	HANDLER_ERROR_ERR(cudaMemcpy(d_a.elements, h_a.elements, ARRAY_BYTES, cudaMemcpyHostToDevice));
	HANDLER_ERROR_ERR(cudaMemcpy(d_b.elements, h_b.elements, ARRAY_BYTES, cudaMemcpyHostToDevice));

	matrixMultiplicationKernel <<<GridBlocks, ThreadsBlocks>>>(d_a, d_b, d_c);
	HANDLER_ERROR_MSG("kernel panic!!!");

	HANDLER_ERROR_ERR(cudaMemcpy(h_c.elements, d_c.elements, ARRAY_BYTES, cudaMemcpyDeviceToHost));

	timer.Stop();

	printf("Time with blocking:  %f ms\n", timer.Elapsed());

	HANDLER_ERROR_ERR(cudaFree(d_a.elements));
	HANDLER_ERROR_ERR(cudaFree(d_b.elements));
	HANDLER_ERROR_ERR(cudaFree(d_c.elements));
}

void test()
{
	Matrix<float> h_a, h_b, h_c;

	h_a.width = N;
	h_a.height = N;

	h_b.width = N;
	h_b.height = N;

	h_c.width = N;
	h_c.height = N;

	h_a.elements = (float*)malloc(h_a.width  * h_b.height * sizeof(int));
	h_b.elements = (float*)malloc(h_b.width  * h_b.height * sizeof(int));
	h_c.elements = (float*)malloc(h_c.width  * h_c.height * sizeof(int));

	int i, j, k = 1;

	for (i = 0; i < h_a.height; i++)
	{
		for (j = 0; j < h_a.height; j++)
		{
			h_a.setElement(i, j, k);
			h_b.setElement(i, j, 1.0);
			h_c.setElement(i, j, 0.0);
			k++;
		}
	}

	onDevice(h_a, h_b, h_c);

	//printf("-: successful execution :-\n");

	system("pause");

	free(h_a.elements);
	free(h_b.elements);
	free(h_c.elements);
}


int main()
{
	test();
}