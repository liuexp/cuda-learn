//FIXME: mutex & cond variable
#include<cstdio>
#include<pthread.h>

typedef struct {
	int size;
	int v;
	int *vs;
} VertexList;

const float eps = 1e-7;
pthread_cond_t tcond;

__global__ void vecAdd(float *C, float *A, float *B, int N){
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if(i<N)C[i] = A[i] + B[i];
}

void testMem(void *x){
	int z = (int) x;
	printf("child here\n");
	pthread_exit(NULL);
}

int main(){
	const int n = (1<<23);
	srand(time(NULL));
	const int ns = n * sizeof(float);
	pthread_t child;
	pthread_attr_t attr;

	pthread_cond_init(&tcond, NULL);
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

	pthread_create(&child, &attr, testMem, (void *)n);

	printf("%d\n",ns);
	//printf("waiting for cond.\n");
	//pthread_mutex_lock(&tcondLock);
	//pthread_cond_wait(&tcond, &tcondLock);
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

	pthread_join(child, NULL);
	printf("child exited.\n");
	vecAdd<<<n/256, 256>>> (dC, dA, dB, n);
	cudaMemcpy(C, dC, ns, cudaMemcpyDeviceToHost);
	for(int i=0;i<100;i++){
		if(abs(C[i] - A[i] - B[i])<eps)continue;
		printf("%.3f + %.3f = %.3f\t %.3f\n", A[i], B[i], C[i], A[i] + B[i]);
	}
	gets(NULL);
	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);

	free(A);
	free(B);
	free(C);

	pthread_attr_destroy(&attr);
	pthread_cond_destroy(&tcond);
	pthread_exit(NULL);
}
