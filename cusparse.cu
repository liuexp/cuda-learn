// Compile with nvcc -arch=sm_20 -O2 -lcusparse cusparse.cu
#include<cstdio>
#include<cstdlib>
#include<ctime>
#include<cuda_runtime.h>
#include<cusparse_v2.h>

//const float RANDRESET = 0.15;
const float DAMPINGFACTOR = 0.85;
const char mtxFile[] = "/media/tmp/graphchi/data/test4";

void FIXLINE(char *s){
	int l = (int)strlen(s) - 1;
	if(s[l] == '\n')s[l] = 0;
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
	int	nnz, n;
	cusparseStatus_t status;

	//time_t t0, t1;
	//double diff;
	clock_t tt;

	n = 23026589;
	nnz = 324874844;

	cooRowHostIdx = (int *) malloc(nnz * sizeof(int));
	cooColHostIdx = (int *) malloc(nnz * sizeof(int));
	cooValHost = (float *) malloc(nnz * sizeof(float));

	readMatrix(cooRowHostIdx, cooColHostIdx, cooValHost, nnz);
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

	printf("starting memcpy\n");
	tt = clock();
	cudaMemcpy(cooRowIdx, cooRowHostIdx, nnz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(cooColIdx, cooColHostIdx, nnz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(cooVal, cooValHost, nnz * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(x, xHost, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(z, zHost, n * sizeof(float), cudaMemcpyHostToDevice);
	printf("memcpy done in %.3fs\n", ((double)clock() - tt)/CLOCKS_PER_SEC);


	cusparseCreate(&handle);
	cusparseCreateMatDescr(&descr); 
	cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);  
	cusparseXcoo2csr(handle,cooRowIdx,nnz,n,csrRow,CUSPARSE_INDEX_BASE_ZERO); 

	const float tmpFloat1 = 1.0;
	
	printf("starting iteration\n");
	tt = clock();
	// for each iteration, y<-z, y<-(1-d)*M*x + y, x <- y
	cudaMemcpy(y, z, nnz * sizeof(float), cudaMemcpyDeviceToDevice);
	status = cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz,
				   &DAMPINGFACTOR, descr, cooVal, csrRow, cooColIdx, x, &tmpFloat1, y);
	if(CUSPARSE_STATUS_SUCCESS != status){ //should use switch case here
		printf("meow\n");
	}
	cudaMemcpy(x, y, n * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaDeviceSynchronize();
	printf("iteration done in %.3fs\n", ((double)clock() - tt)/CLOCKS_PER_SEC);

	printf("starting copying to host\n");
	tt = clock();
	cudaMemcpy(xHost, x, n * sizeof(float), cudaMemcpyDeviceToHost);
	printf("memcpy done in %.3fs\n", ((double)clock() - tt)/CLOCKS_PER_SEC);

	for(int i=0;i<10;i++){
		printf("%d\t%.3f\n", i, x[i]);
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
	return 0;
}

