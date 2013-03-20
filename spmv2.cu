#include<cstdio>
#include<cstdlib>
#include<ctime>
#include<cuda_runtime.h>
#include<cassert>
#include "common.h"

//y<-alpha*A*x+z
template <typename IndexType, typename ValueType>
__global__ void
spmv_csr_scalar_kernel(IndexType numRows, IndexType *csrRow, IndexType *cooColIdx, int *outDegree, ValueType *x, ValueType *y, ValueType alpha, ValueType beta)
{
	const IndexType thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	const IndexType grid_size = gridDim.x * blockDim.x;
	//FIXME: x[col]/outDegree[col] should be done first
	for(IndexType row = thread_id; row < numRows; row += grid_size)
	{
		const IndexType row_start = csrRow[row];
		const IndexType row_end   = csrRow[row+1];
		
		ValueType sum = 0;
		for (IndexType jj = row_start; jj < row_end; jj++){
			IndexType col = cooColIdx[jj];
			sum += x[col] / ((float)outDegree[col]);
		}
		
		y[row] = alpha * sum + beta;
	}
}

void spmv_csr_scalar(int numRows, int *csrRow, int *cooColIdx, int *outDegree, float *x, float *y, float alpha, float beta)
{
	const size_t BLOCK_SIZE = 256;
	int T_BLOCKS = (int)DIVIDE_INTO(numRows, BLOCK_SIZE);
	const size_t MAX_BLOCKS = max_active_blocks(spmv_csr_scalar_kernel<int, float>, BLOCK_SIZE, (size_t) 0);
	const size_t NUM_BLOCKS = min((int)MAX_BLOCKS, T_BLOCKS);
	spmv_csr_scalar_kernel<int, float> <<<NUM_BLOCKS, BLOCK_SIZE>>> 
	    (numRows, csrRow, cooColIdx, outDegree, x, y, alpha, beta);
}

int main(){
	int	*csrHost, *cooColHostIdx;
	int	*outDegreeHost;
	float	*xHost, *yHost;

	tt0 = clock();
	time(&realt0);
	
	csrHost = (int *) malloc(n * sizeof(int));
	cooColHostIdx = (int *) malloc(nnz * sizeof(int));
	outDegreeHost = (int *) malloc(n * sizeof(float));

	readMetaMatrix(outDegreeHost, NULL, csrHost);

	xHost = (float *) malloc(n * sizeof(float));
	yHost = (float *) malloc(n * sizeof(float));
	for(int i=0;i<n;i++)xHost[i] = 1.0;
	
	int	*cooRowIdx, *cooColIdx;
	float	*cooVal;
	int	*csrRow;
	float	*x, *y;

	const unsigned int maxNNZPerTurn = min(50000000,nnz);
	const unsigned int maxNPerTurn = min(maxNNZPerTurn, n);

	cooRowIdx = NULL;
	handleError(cudaMalloc((void **)&cooColIdx, maxNNZPerTurn * sizeof(int)));
	handleError(cudaMalloc((void **)&cooVal, maxNNZPerTurn * sizeof(float)));
	handleError(cudaMalloc((void **)&x, n * sizeof(float)));
	handleError(cudaMalloc((void **)&y, maxNPerTurn * sizeof(float)));
	handleError(cudaMalloc((void**)&csrRow, (maxNPerTurn+1)*sizeof(int)));
	reportTime(tt0);
	reportTimeReal();

	//FIXME: change the pipeline for the in-memory CSR
	int *csrRowHost = (int *) malloc(sizeof(int) * (maxNPerTurn + 1));

	// starting block operation
	// for now we group edges
	for(int iter = 0;iter<niter;iter++){
		printf("---------------\n");
		printf("iteration %d starting:\n", iter);
		clock_t t_iter = clock();
		
		memset(yHost, 0, sizeof(float)*n);
		unsigned int nnzCurTurn = loadBlockMatrixCoo(cooColHostIdx, cooRowHostIdx, cooValHost, 0);
		unsigned int nCurTurn = matCoo2Csr(cooColHostIdx, cooRowHostIdx, csrRowHost, nnzCurTurn);
		//TODO:change memcpy to async
		handleError(cudaMemcpy(csrRow, csrRowHost, sizeof(int) * (nCurTurn + 1), cudaMemcpyHostToDevice));
		handleError(cudaMemcpy(cooColIdx, cooColHostIdx, nnzCurTurn * sizeof(int), cudaMemcpyHostToDevice));
		handleError(cudaMemcpy(cooVal, cooValHost, nnzCurTurn * sizeof(float), cudaMemcpyHostToDevice));
		handleError(cudaMemcpy(x, xHost, n * sizeof(float), cudaMemcpyHostToDevice));
		int lastRow = -1;
		int lastPartialResult = 0;
		int curXOffset = 0;

		for(int i = 1; i < numShards - 1; i++){
			printf("[Turn %d] started.\n", i);
			spmv_csr_scalar(nCurTurn, csrRow, cooColIdx, cooVal, x, y, DAMPINGFACTOR, (1-DAMPINGFACTOR)/n);
			if(lastRow == curXOffset)
				lastPartialResult = yHost[lastRow];
			handleError(cudaMemcpy(yHost + curXOffset, y, nCurTurn * sizeof(float), cudaMemcpyDeviceToHost));
			curXOffset += nCurTurn;
			nnzCurTurn = loadBlockMatrixCoo(cooColHostIdx, cooRowHostIdx, cooValHost, 0);
			nCurTurn = matCoo2Csr(cooColHostIdx, cooRowHostIdx, csrRowHost, nnzCurTurn);
			handleError(cudaMemcpy(csrRow, csrRowHost, sizeof(int) * (nCurTurn + 1), cudaMemcpyHostToDevice));
			handleError(cudaMemcpy(cooColIdx, cooColHostIdx, nnzCurTurn * sizeof(int), cudaMemcpyHostToDevice));
			handleError(cudaMemcpy(cooVal, cooValHost, nnzCurTurn * sizeof(float), cudaMemcpyHostToDevice));
			//FIXME: only need to wait for yHost's copy is complete.
			cudaDeviceSynchronize();
			yHost[lastRow] += lastPartialResult;
			lastRow = curXOffset + nCurTurn -1;
			reportTimeReal();
		}

		spmv_csr_scalar(nCurTurn, csrRow, cooColIdx, cooVal, x, y, DAMPINGFACTOR, (1-DAMPINGFACTOR)/n);
		if(lastRow == curXOffset)
			lastPartialResult = yHost[lastRow];
		handleError(cudaMemcpy(yHost + curXOffset, y, nCurTurn * sizeof(float), cudaMemcpyDeviceToHost));
		curXOffset += nCurTurn;
		cudaDeviceSynchronize();
		yHost[lastRow] += lastPartialResult;

		memcpy(xHost, yHost, n*sizeof(float));
		reportTimeRound("iteration",t_iter);
		reportTime(tt0);
		reportTimeReal();
	}
	dumpRes(xHost);

	free(xHost);
	free(yHost);
	free(cooColHostIdx);
	free(csrHost);
	free(outDegreeHost);
	
	cudaFree(cooColIdx);
	cudaFree(cooRowIdx);
	cudaFree(cooVal);
	cudaFree(csrRow);
	cudaFree(x);
	cudaFree(y);

	reportTime(tt0);
	reportTimeReal();
	cudaDeviceReset();
	return 0;
}

