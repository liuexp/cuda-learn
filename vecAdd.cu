#include<cstdio>

typedef struct {
	int size;
	int v;
	int *vs;
} VertexList;

const float eps = 1e-7;

__global__ void vecAdd(float *C, float *A, float *B, int N){
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if(i<N)C[i] = A[i] + B[i];
}

int main(){
	const int n = (1<<28);
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
	for(int i=0;i<n;i++){
		A[i]=float(rand())/RAND_MAX;
		B[i]=float(rand())/RAND_MAX;
	}
	printf("init done\n");
	cudaMemcpy(dA, A, ns, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, B, ns, cudaMemcpyHostToDevice);
	dim3 block(512);
	dim3 grid;
	int TBLOCKS = n/block.x;
	grid.x = TBLOCKS % 65536;
	grid.y = TBLOCKS / 65536 + 1;

	vecAdd<<<grid, block>>> (dC, dA, dB, n);
	cudaMemcpy(C, dC, ns, cudaMemcpyDeviceToHost);
	for(int i=0;i<n;i++){
		if(abs(C[i] - A[i] - B[i])<eps)continue;
		printf("err at %d\n", i);
		printf("%.3f + %.3f = %.3f\t %.3f\n", A[i], B[i], C[i], A[i] + B[i]);
		break;
	}
	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);

	free(A);
	free(B);
	free(C);
	getchar();
}
