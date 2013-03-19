#include<cstdio>
#include "common.h"

typedef struct {
	int size;
	int v;
	int *vs;
} VertexList;

const float eps = 1e-7;

__global__ void vecAdd(float *C, float *A, float *B, int N){
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	int gridSize = gridDim.x * blockDim.x;

	for(int i=id;i<N;i+=gridSize)
		C[i] = A[i] + B[i];
}

int main(){
	const int n = (1<<30)/3;
	srand(time(NULL));
	const unsigned int ns = n * sizeof(float);

	printf("%u\n",ns);
	float *A,*B,*C,*dA,*dB,*dC;
	A=(float *)malloc(ns);
	B=(float *)malloc(ns);
	C=(float *)malloc(ns);
	handleError(cudaMalloc((void **)&dA, ns));
	handleError(cudaMalloc((void **)&dB, ns));
	handleError(cudaMalloc((void **)&dC, ns));
	for(int i=0;i<n;i++){
		A[i]=float(rand())/RAND_MAX;
		B[i]=float(rand())/RAND_MAX;
	}
	printf("init done\n");
	cudaMemcpy(dA, A, ns, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, B, ns, cudaMemcpyHostToDevice);
	dim3 block(256);
	dim3 grid;
	int TBLOCKS = n/block.x;
	grid.x = TBLOCKS % 65535;
	grid.y = TBLOCKS / 65535 + 1;

	vecAdd<<<grid, block>>> (dC, dA, dB, n);
	cudaMemcpy(C, dC, ns, cudaMemcpyDeviceToHost);
	for(int i=0;i<n;i++){
		if(abs(C[i] - A[i] - B[i])<eps)continue;
		printf("err at %d\n", i);
		printf("%.3f + %.3f = %.3f\t %.3f\n", A[i], B[i], C[i], A[i] + B[i]);
		break;
	}
	printf("done.\n");
	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);

	free(A);
	free(B);
	free(C);
	getchar();
}
