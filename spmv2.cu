#include<cstdio>
#include<cstdlib>
#include<ctime>
#include<cuda_runtime.h>
#include<cuda.h>
#include<cassert>
#include "common.h"
//#include <helper_cuda.h>
//#include <helper_functions.h> 

#define STREAM_COUNT 4
cudaEvent_t cycleDone[STREAM_COUNT];
cudaStream_t stream[STREAM_COUNT];
cudaEvent_t start, stop;

//y<-alpha*A*x+z
template <typename IndexType, typename ValueType>
__global__ void
spmv_csr_scalar_kernel(IndexType numRows, IndexType cooOffset, IndexType *csrRow, IndexType *cooColIdx, int *outDegree, ValueType *x, ValueType *y, ValueType alpha, ValueType beta)
{
	const IndexType thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	const IndexType grid_size = gridDim.x * blockDim.x;
	//FIXME: x[col]/outDegree[col] should be done before aggregate
	for(IndexType row = thread_id; row < numRows; row += grid_size)
	{
		if(csrRow[row] < cooOffset)continue;
		const IndexType row_start = csrRow[row] - cooOffset; 	//NOTE: row_start can be unsigned so may never < 0
		const IndexType row_end   = csrRow[row+1] - cooOffset;
		
		ValueType sum = 0;
		for (IndexType jj = row_start; jj < row_end; jj++){
			IndexType col = cooColIdx[jj];
			sum += x[col] / ((float)outDegree[col]);
		}
		
		y[row] = alpha * sum + beta;
		//FIXME: +beta should be done after aggregate
	}
}

void spmv_csr_scalar(int numRows, int cooOffset, int *csrRow, int *cooColIdx, int *outDegree, float *x, float *y, float alpha, float beta)
{
	const size_t BLOCK_SIZE = 512;
	int T_BLOCKS = (int)DIVIDE_INTO(numRows, BLOCK_SIZE);
	const size_t MAX_BLOCKS = max_active_blocks(spmv_csr_scalar_kernel<int, float>, BLOCK_SIZE, (size_t) 0);
	const size_t NUM_BLOCKS = min((int)MAX_BLOCKS, T_BLOCKS);
	spmv_csr_scalar_kernel<int, float> <<<NUM_BLOCKS, BLOCK_SIZE>>> 
	    (numRows, cooOffset, csrRow, cooColIdx, outDegree, x, y, alpha, beta);
}

int main(){
	//TODO: cooColHostIdx[STREAM_COUNT];
	int	*csrHost, *cooColHostIdx;
	int	*outDegreeHost;
	float	*xHost, *yHost;

	tt0 = clock();
	time(&realt0);
	
	readMetaMatrix(&outDegreeHost, NULL, &csrHost);
	//cooColHostIdx = (int *) malloc(nnz * sizeof(int));
	handleError(cudaHostAlloc(&cooColHostIdx, nnz*sizeof(int), cudaHostAllocDefault));
	if(cooColHostIdx == NULL)exit(-1);


	//xHost = (float *) malloc(n * sizeof(float));
	//yHost = (float *) malloc(n * sizeof(float));
	handleError(cudaHostAlloc(&xHost, nnz*sizeof(float), cudaHostAllocDefault));
	handleError(cudaHostAlloc(&yHost, nnz*sizeof(float), cudaHostAllocDefault));
	for(unsigned int i=0;i<n;i++)yHost[i] = 1.0/n;
	
	cudaEventCreate(&start);
	cudaEventCreate(&stop);


	//TODO: cooColIdx[STREAM_COUNT];
	int	*cooColIdx;
	int	*csr;
	int	*outDegree;
	float	*x, *y;

	const unsigned int maxNNZPerTurn = min(GPUMEM,nnz);
	const unsigned int maxNPerTurn = min(maxNNZPerTurn, n);

	handleError(cudaMalloc((void **)&cooColIdx, maxNNZPerTurn * sizeof(int)));
	handleError(cudaMalloc((void **)&outDegree, n * sizeof(int)));
	handleError(cudaMalloc((void **)&x, n * sizeof(float)));
	handleError(cudaMalloc((void **)&y, maxNPerTurn * sizeof(float)));
	handleError(cudaMalloc((void**)&csr, (n+1)*sizeof(int)));

	handleError(cudaMemcpy(csr, csrHost, sizeof(int) * (n + 1), cudaMemcpyHostToDevice));
	handleError(cudaMemcpy(outDegree, outDegreeHost, n * sizeof(int), cudaMemcpyHostToDevice));
	reportTime(tt0);
	reportTimeReal();
	// starting block operation
	// for now we group edges
	for(int iter = 0;iter<niter;iter++){
		printf("---------------\n");
		printf("iteration %d starting:\n", iter);
		clock_t t_iter = clock();
		
		int nCurTurn, cooOffset;
		unsigned int nnzCurTurn = loadBlockMatrixCsr(cooColHostIdx, 0, nCurTurn, cooOffset);
		handleError(cudaMemcpyAsync(cooColIdx, cooColHostIdx, nnzCurTurn * sizeof(int), cudaMemcpyHostToDevice, 0));
		handleError(cudaMemcpyAsync(x, yHost, n * sizeof(float), cudaMemcpyHostToDevice, 0));
		int lastRow = -1;
		int lastPartialResult = 0;
		int curXOffset = 0;

		for(unsigned int i = 1; i < numShards ; i++){
			printf("[Turn %d] started.\n", i);
			int csrOffset = cooOffset >= csrHost[lastRow + 1] ? lastRow + 1: lastRow;
			cudaEventRecord(start,0);
			//FIXME:specify a stream
			spmv_csr_scalar(nCurTurn, cooOffset, &csr[csrOffset], cooColIdx, outDegree, x, y, DAMPINGFACTOR, (1-DAMPINGFACTOR)/n);
			if(lastRow == curXOffset)
				lastPartialResult = yHost[lastRow];
			handleError(cudaMemcpyAsync(&yHost[curXOffset], y, nCurTurn * sizeof(float), cudaMemcpyDeviceToHost, 0));
			cudaEventRecord(stop,0);
			curXOffset += nCurTurn;
			nnzCurTurn = loadBlockMatrixCsr(cooColHostIdx, i, nCurTurn, cooOffset);
			handleError(cudaMemcpyAsync(cooColIdx, cooColHostIdx, nnzCurTurn * sizeof(int), cudaMemcpyHostToDevice, 0));
			cudaDeviceSynchronize();
			//cudaEventSynchronize(stop);
			yHost[lastRow] += lastPartialResult - (1-DAMPINGFACTOR)/n;
			lastRow = curXOffset + nCurTurn -1;
			reportTimeReal();
		}

		int csrOffset = cooOffset >= csrHost[lastRow + 1] ? lastRow + 1: lastRow;
		spmv_csr_scalar(nCurTurn, cooOffset, &csr[csrOffset], cooColIdx, outDegree, x, y, DAMPINGFACTOR, (1-DAMPINGFACTOR)/n);
		if(lastRow == curXOffset)
			lastPartialResult = yHost[lastRow];
		handleError(cudaMemcpyAsync(&yHost[curXOffset] , y, nCurTurn * sizeof(float), cudaMemcpyDeviceToHost, 0));
		curXOffset += nCurTurn;
		cudaDeviceSynchronize();
		yHost[lastRow] += lastPartialResult - (1-DAMPINGFACTOR)/n;

		//memcpy(xHost, yHost, n*sizeof(float));
		reportTimeRound("iteration",t_iter);
		reportTime(tt0);
		reportTimeReal();
	}
	//FIXME: xHost is not needed.
	memcpy(xHost, yHost, n*sizeof(float));
	dumpRes(xHost);

//	free(xHost);
//	free(yHost);
//	free(cooColHostIdx);
	cudaFreeHost(xHost);
	cudaFreeHost(yHost);
	cudaFreeHost(cooColHostIdx);
	free(csrHost);
	free(outDegreeHost);
	
	cudaFree(cooColIdx);
	cudaFree(outDegree);
	cudaFree(csr);
	cudaFree(x);
	cudaFree(y);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);


	reportTime(tt0);
	reportTimeReal();
	cudaDeviceReset();
	return 0;
}

