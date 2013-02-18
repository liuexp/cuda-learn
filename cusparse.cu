// Compile with nvcc -arch=sm_20 -O2 -lcusparse cusparse.cu
#include<cstdio>
#include<cstdlib>
#include<ctime>
#include<cuda_runtime.h>
#include<cusparse_v2.h>

//const float RANDRESET = 0.15;
const float DAMPINGFACTOR = 0.85;
const char mtxFile[] = "/media/tmp/graphchi/data/test3";
const int n = 61578171;
const int nnz = 345439900;
//const int n = 4, nnz = 9;
const int niter = 4;

void handleError(cudaError_t z){
	switch(z){
		case cudaErrorInvalidDevicePointer:
			printf("invalid device ptr\n");
			break;
		case cudaErrorInvalidSymbol:
			printf("invalid symbol\n");
			break;
		case cudaErrorMemoryAllocation:
			printf("failed mem alloc\n");
			break;
		case cudaErrorMixedDeviceExecution:
			printf("mixed device execution\n");
			break;
		case cudaSuccess:
			printf("success memcpy\n");
			break;
		default:
			printf("unknown\n");
			break;
	}

}

void FIXLINE(char *s){
	int l = (int)strlen(s) - 1;
	if(s[l] == '\n')s[l] = 0;
}

void readSampleMatrix(int *row, int *col, float *val, int m){
	row[0]=0; col[0]=0; val[0]=1.0;  
	row[1]=0; col[1]=2; val[1]=2.0;  
	row[2]=0; col[2]=3; val[2]=3.0;  
	row[3]=1; col[3]=1; val[3]=4.0;  
	row[4]=2; col[4]=0; val[4]=5.0;  
	row[5]=2; col[5]=2; val[5]=6.0;
	row[6]=2; col[6]=3; val[6]=7.0;  
	row[7]=3; col[7]=1; val[7]=8.0;  
	row[8]=3; col[8]=3; val[8]=9.0;  
}

void readMatrix(int *row, int *col, float *val, int m){
	FILE *fp = fopen(mtxFile,"r");
	char s[1024];
	int cnt = 0;
	clock_t tt = clock();
	while(fgets(s, 1024, fp)){
		FIXLINE(s);
		char del[] = "\t ";
		if(s[0] == '#' || s[0] == '%')
			continue;
		char *t;
		t = strtok(s, del);
		int a,b;
		float c;
		a = atoi(t);
		t = strtok(NULL, del);
		b = atoi(t);
		t = strtok(NULL,del);
		c = atof(t);
		row[cnt] = a;
		col[cnt] = b;
		val[cnt] = c;
		cnt++;
	}
	printf("Read %d lines matrix in %.3fs\n", cnt, ((double)clock() - tt)/CLOCKS_PER_SEC);
	fclose(fp);
}

int main(){
	cusparseHandle_t handle=0;
	cusparseMatDescr_t descr=0;
	int	*cooRowHostIdx, *cooColHostIdx;
	float	*cooValHost;
	int	*cooRowIdx, *cooColIdx;
	float	*cooVal;
	int	*csrRow;
	float	*xHost, *zHost;
	float	*x, *y, *z;
	cusparseStatus_t status;
	cudaError_t cudaStat;

	clock_t tt,tt0;
	tt0 = clock();

	cooRowHostIdx = (int *) malloc(nnz * sizeof(int));
	cooColHostIdx = (int *) malloc(nnz * sizeof(int));
	cooValHost = (float *) malloc(nnz * sizeof(float));

	readMatrix(cooRowHostIdx, cooColHostIdx, cooValHost, nnz);
	//readSampleMatrix(cooRowHostIdx, cooColHostIdx, cooValHost, nnz);
	xHost = (float *) malloc(n * sizeof(float));
	//yHost = (float *) malloc(n * sizeof(float));
	zHost = (float *) malloc(n * sizeof(float)); // the constant vector
	for(int i=0;i<n;i++)xHost[i] = 1.0;
	for(int i=0;i<n;i++)zHost[i] = (1 - DAMPINGFACTOR) / n;

	cudaMalloc((void **)&cooRowIdx, nnz * sizeof(int));
	cudaMalloc((void **)&cooColIdx, nnz * sizeof(int));
	cudaMalloc((void **)&cooVal, nnz * sizeof(float));
	cudaMalloc((void **)&x, n * sizeof(float));
	cudaMalloc((void **)&y, n * sizeof(float));
	cudaMalloc((void **)&z, n * sizeof(float));
	cudaMalloc((void**)&csrRow,(n+1)*sizeof(csrRow[0]));
	printf("-- ELAPSED TIME: %.3fs\n", ((double)clock() - tt0)/CLOCKS_PER_SEC);

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
	cudaStat = cudaMemcpy(z, zHost, n * sizeof(float), cudaMemcpyHostToDevice);
	handleError(cudaStat);
	cudaStat = cudaMemcpy(y, zHost, n * sizeof(float), cudaMemcpyHostToDevice);
	handleError(cudaStat);
	cudaDeviceSynchronize();
	printf("memcpy done in %.3fs\n", ((double)clock() - tt)/CLOCKS_PER_SEC);
	printf("-- ELAPSED TIME: %.3fs\n", ((double)clock() - tt0)/CLOCKS_PER_SEC);

	cusparseCreate(&handle);
	cusparseCreateMatDescr(&descr); 
	cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);  
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
					printf("unknown\n");
				break;
		}
		if(CUSPARSE_STATUS_SUCCESS != status){ //should use switch case here
			printf("meow\n");
		}
		cudaStat = cudaMemcpy(x, y, n * sizeof(float), cudaMemcpyDeviceToDevice);
		handleError(cudaStat);
		cudaDeviceSynchronize();
		printf("iteration done in %.6fs\n", ((double)clock() - tt)/CLOCKS_PER_SEC);
		printf("-- ELAPSED TIME: %.6fs\n", ((double)clock() - tt0)/CLOCKS_PER_SEC);
	}
	printf("starting copying to host\n");
	tt = clock();
	cudaDeviceSynchronize();
	//FIXME:taking from y is probably faster than taking from x....
	cudaStat = cudaMemcpy(xHost, y, n * sizeof(float), cudaMemcpyDeviceToHost);
	handleError(cudaStat);
	printf("memcpy done in %.3fs\n", ((double)clock() - tt)/CLOCKS_PER_SEC);
	printf("-- ELAPSED TIME: %.3fs\n", ((double)clock() - tt0)/CLOCKS_PER_SEC);

	for(int i=0;i<min(n,10);i++){
		printf("%d\t%.9f\t%.9f\n", i, xHost[i], zHost[i]);
	}

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
	printf("-- ELAPSED TIME: %.3fs\n", ((double)clock() - tt0)/CLOCKS_PER_SEC);

	/* destroy matrix descriptor */ 
	status = cusparseDestroyMatDescr(descr); 
	descr = 0;
	if (status != CUSPARSE_STATUS_SUCCESS) {
	    printf("Matrix descriptor destruction failed");
	    return 1;
	}    
	
	/* destroy handle */
	status = cusparseDestroy(handle);
	handle = 0;
	if (status != CUSPARSE_STATUS_SUCCESS) {
	    printf("CUSPARSE Library release of resources failed");
	    return 1;
	}   
	cudaDeviceReset();
	return 0;
}

