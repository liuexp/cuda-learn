#include<cstdio>
#include "common.h"

typedef struct {
	int size;
	int v;
	int *vs;
} VertexList;

const float eps = 1e-7;

__global__ void vecAdd(float *A, float *B, int N){
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	int gridSize = gridDim.x * blockDim.x;

	for(int i=id;i<N;i+=gridSize)
		A[i] = A[i] + B[i];
}

int main(){
	const int n = (1<<29);
	srand(time(NULL));
	const unsigned int ns = n * sizeof(float);

//	cudaThreadExit();
//	cudaSetDevice(1);


	printf("%u\n",ns);
	float *A,*B,*C,*dA,*dB;
	A=(float *)malloc(ns);
	B=(float *)malloc(ns);
	C=(float *)malloc(ns);
	handleError(cudaMalloc((void **)&dA, ns));
	handleError(cudaMalloc((void **)&dB, ns));
	for(int i=0;i<n;i++){
		A[i]=float(rand())/RAND_MAX;
		B[i]=float(rand())/RAND_MAX;
	}
	printf("init done\n");
	cudaMemcpy(dA, A, ns, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, B, ns, cudaMemcpyHostToDevice);

	const size_t BLOCK_SIZE = 256;
	int T_BLOCKS = (int)DIVIDE_INTO(n, BLOCK_SIZE);
	const size_t MAX_BLOCKS = max_active_blocks(vecAdd, BLOCK_SIZE, (size_t) 0);
	const size_t NUM_BLOCKS = min((int)MAX_BLOCKS, T_BLOCKS);
	vecAdd<<<NUM_BLOCKS, BLOCK_SIZE>>> (dA, dB, n);
	cudaMemcpy(C, dA, ns, cudaMemcpyDeviceToHost);
	for(int i=0;i<n;i++){
		if(abs(C[i] - A[i] - B[i])<eps)continue;
		printf("err at %d\n", i);
		printf("%.3f + %.3f = %.3f\t %.3f\n", A[i], B[i], C[i], A[i] + B[i]);
		break;
	}
	printf("done.\n");
	cudaFree(dA);
	cudaFree(dB);

	free(A);
	free(B);
	free(C);
	getchar();
	cudaDeviceReset();
}
