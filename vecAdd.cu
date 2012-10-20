#include<cstdio>

__global__ void vecAdd(float *C, float *A, float *B){
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	C[i] = A[i] + B[i];
}

int main(){
	const int n = (1<<15);
	srand(time(NULL));
	const int ns = n * sizeof(float);

	printf("%d\n",ns);
	float *A,*B,*C,*dA,*dB,*dC;
	A=(float *)malloc(ns);
	B=(float *)malloc(ns);
	C=(float *)malloc(ns);
	cudaMalloc((void **)&dA, ns);
	cudaMalloc((void **)&dB, ns);
	cudaMalloc((void **)&dC, ns);
	for(int i=0;i<1000;i++){
		A[i]=float(rand())/RAND_MAX;
		B[i]=float(rand())/RAND_MAX;
	}
	cudaMemcpy(dA, A, ns, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, B, ns, cudaMemcpyHostToDevice);
	vecAdd<<<ns/256, 256>>> (dC, dA, dB);
	cudaMemcpy(C, dC, ns, cudaMemcpyDeviceToHost);
	for(int i=0;i<100;i++){
		printf("%.3f + %.3f = %.3f\n", A[i], B[i], C[i]);
	}
	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);

	free(A);
	free(B);
	free(C);
}
