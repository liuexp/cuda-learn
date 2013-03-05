// Compile with nvcc -arch=sm_20 -O2 -lcusparse cusparse.cu
//FIXME: what if n is too large for one turn?
#include<cstdio>
#include<cstdlib>
#include<ctime>
#include<cuda_runtime.h>
#include<cusparse_v2.h>
#include "common.h"

void handleStatus(cusparseStatus_t status){
	switch(status){
		case CUSPARSE_STATUS_INVALID_VALUE:
			printf("invalid value");
			break;
		case CUSPARSE_STATUS_NOT_INITIALIZED:
			printf("not initialized");
			break;
		case CUSPARSE_STATUS_ARCH_MISMATCH:
			printf("arch mismatch");
			break;
		case CUSPARSE_STATUS_EXECUTION_FAILED:
			printf("exe failed");
			break;
		case CUSPARSE_STATUS_INTERNAL_ERROR:
			printf("internal error");
			break;
		case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
			printf("not supported");
			break;
		case CUSPARSE_STATUS_ALLOC_FAILED:
			printf("alloc failed");
			break;
		case CUSPARSE_STATUS_MAPPING_ERROR :
			printf("map error");
			break;
		case CUSPARSE_STATUS_SUCCESS:
			printf("success\n");
			break;
		default:
				printf("unknown status\n");
			break;
	}
}

//FIXME:give n and nnz another name
//FIXME:iteration should be outside of turns?
void matMult(int *cooRowHostIdx, int *cooColHostIdx, float *cooValHost, float *xHost, float *zHost, int n, int nnz, 
		int *cooRowIdx, int *cooColIdx, float *cooVal, float *x, float *y, float *z, int *csrRow,
		int xOffset, int nPerTurn){
	clock_t tt;
	cusparseStatus_t status;
	cudaError_t cudaStat;
	cusparseHandle_t handle=0;
	cusparseMatDescr_t descr=0;
	printf("starting memcpy to device\n");
	tt = clock();
	cudaStat = cudaMemcpy(cooRowIdx, cooRowHostIdx, nnz * sizeof(int), cudaMemcpyHostToDevice);
	handleError(cudaStat);
	cudaStat = cudaMemcpy(cooColIdx, cooColHostIdx, nnz * sizeof(int), cudaMemcpyHostToDevice);
	handleError(cudaStat);
	cudaStat = cudaMemcpy(cooVal, cooValHost, nnz * sizeof(float), cudaMemcpyHostToDevice);
	handleError(cudaStat);
	cudaStat = cudaMemcpy(x, xHost, n * sizeof(float), cudaMemcpyHostToDevice);
	handleError(cudaStat);
	cudaStat = cudaMemcpy(z, zHost, nPerTurn * sizeof(float), cudaMemcpyHostToDevice);
	handleError(cudaStat);
	cudaStat = cudaMemcpy(y, zHost, nPerTurn * sizeof(float), cudaMemcpyHostToDevice);
	handleError(cudaStat);
	cudaDeviceSynchronize();
	reportTimeRound("memcpy", tt);
	reportTime(tt0);

	cusparseCreate(&handle);
	cusparseCreateMatDescr(&descr); 
	cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);  
	//FIXME:conversion
	cusparseXcoo2csr(handle,cooRowIdx,nnz,n,csrRow,CUSPARSE_INDEX_BASE_ZERO); 

	const float tmpFloat1 = 1.0;
	printf("starting iteration\n");
	for(int i=0;i<niter;i++){
		tt = clock();
		// for each iteration, y<-z, y<-(1-d)*M*x + y, x <- y
		cudaStat = cudaMemcpy(y, z, n * sizeof(float), cudaMemcpyDeviceToDevice);
		handleError(cudaStat);
		status = cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz,
					   &DAMPINGFACTOR, descr, cooVal, csrRow, cooColIdx, x, &tmpFloat1, y);
		handleStatus(status);
		if(CUSPARSE_STATUS_SUCCESS != status){ 
			printf("meow\n");
		}
		cudaStat = cudaMemcpy(x, y, n * sizeof(float), cudaMemcpyDeviceToDevice);
		handleError(cudaStat);
		cudaDeviceSynchronize();
		reportTimeRound("iteration",tt);
		reportTime(tt0);
	}
	printf("starting copying to host\n");
	tt = clock();
	cudaDeviceSynchronize();
	//FIXME:taking from y is probably faster than taking from x....
	cudaStat = cudaMemcpy(xHost, y, n * sizeof(float), cudaMemcpyDeviceToHost);
	handleError(cudaStat);

	reportTimeRound("memcpy", tt);
	reportTime(tt0);

	/* destroy matrix descriptor */ 
	status = cusparseDestroyMatDescr(descr); 
	descr = 0;
	if (status != CUSPARSE_STATUS_SUCCESS) {
	    printf("Matrix descriptor destruction failed");
	}    
	
	/* destroy handle */
	status = cusparseDestroy(handle);
	handle = 0;
	if (status != CUSPARSE_STATUS_SUCCESS) {
	    printf("CUSPARSE Library release of resources failed");
	}   
}
int main(){
	int	*cooRowHostIdx, *cooColHostIdx;
	float	*cooValHost;
	float	*xHost, *zHost;
	tt0 = clock();

	cooRowHostIdx = (int *) malloc(nnz * sizeof(int));
	cooColHostIdx = (int *) malloc(nnz * sizeof(int));
	cooValHost = (float *) malloc(nnz * sizeof(float));

	readBinMatrix(cooRowHostIdx, cooColHostIdx, cooValHost, nnz);
	//readMatrix(cooRowHostIdx, cooColHostIdx, cooValHost, nnz);
	//readSampleMatrix(cooRowHostIdx, cooColHostIdx, cooValHost, nnz);
	xHost = (float *) malloc(n * sizeof(float));
	//yHost = (float *) malloc(n * sizeof(float));
	zHost = (float *) malloc(n * sizeof(float)); // the constant vector
	for(int i=0;i<n;i++)xHost[i] = 1.0;
	for(int i=0;i<n;i++)zHost[i] = (1 - DAMPINGFACTOR) / n;
	
	int	*cooRowIdx, *cooColIdx;
	float	*cooVal;
	int	*csrRow;
	float	*x, *y, *z;
	cudaError_t cudaStat;

	int maxNPerTurn = n;
	int maxNNZPerTurn = nnz;

	cudaStat = cudaMalloc((void **)&cooRowIdx, maxNNZPerTurn * sizeof(int));
	handleError(cudaStat);
	cudaStat = cudaMalloc((void **)&cooColIdx, maxNNZPerTurn * sizeof(int));
	handleError(cudaStat);
	cudaStat = cudaMalloc((void **)&cooVal, maxNNZPerTurn * sizeof(float));
	handleError(cudaStat);
	cudaStat = cudaMalloc((void **)&x, n * sizeof(float));
	handleError(cudaStat);
	cudaStat = cudaMalloc((void **)&y, maxNPerTurn * sizeof(float));
	handleError(cudaStat);
	cudaStat = cudaMalloc((void **)&z, maxNPerTurn * sizeof(float));
	handleError(cudaStat);
	cudaStat = cudaMalloc((void**)&csrRow, (n+1)*sizeof(csrRow[0]));
	reportTime(tt0);


	int nCurTurn = n;
	int nnzCurTurn = nnz;

	matMult(cooRowHostIdx, cooColHostIdx, cooValHost, xHost, zHost, n, nnzCurTurn,
			cooRowIdx, cooColIdx, cooVal, x, y, z, csrRow,
			0, nCurTurn);

	dumpRes(xHost);

	free(xHost);
	free(zHost);
	free(cooColHostIdx);
	free(cooRowHostIdx);
	free(cooValHost);
	
	cudaFree(cooColIdx);
	cudaFree(cooRowIdx);
	cudaFree(cooVal);
	cudaFree(csrRow);
	cudaFree(x);
	cudaFree(y);
	cudaFree(z);


	reportTime(tt0);
	cudaDeviceReset();
	return 0;
}

