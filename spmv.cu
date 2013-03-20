//FIXME: what if n is too large for one turn?
#include<cstdio>
#include<cstdlib>
#include<ctime>
#include<cuda_runtime.h>
#include<cassert>
#include "common.h"

//const float eps = 1e-9;
//y<-alpha*A*x+z
template <typename IndexType, typename ValueType>
__global__ void
spmv_csr_scalar_kernel(IndexType numRows, IndexType *csrRow, IndexType *cooColIdx, ValueType *cooVal, ValueType *x, ValueType *y, ValueType alpha, ValueType beta)
{
	const IndexType thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	const IndexType grid_size = gridDim.x * blockDim.x;
	
	for(IndexType row = thread_id; row < numRows; row += grid_size)
	{
		const IndexType row_start = csrRow[row];
		const IndexType row_end   = csrRow[row+1];
		
		ValueType sum = 0;
		for (IndexType jj = row_start; jj < row_end; jj++)
			sum += cooVal[jj] * x[cooColIdx[jj]];       
		y[row] = alpha * sum + beta;
		
	}
}

void spmv_csr_scalar(int numRows, int *csrRow, int *cooColIdx, float *cooVal, float *x, float *y, float alpha, float beta)
{
	const size_t BLOCK_SIZE = 256;
	const size_t MAX_BLOCKS = max_active_blocks(spmv_csr_scalar_kernel<int, float>, BLOCK_SIZE, (size_t) 0);
	int T_BLOCKS = (int)DIVIDE_INTO(numRows, BLOCK_SIZE);
	const size_t NUM_BLOCKS = min((int)MAX_BLOCKS, T_BLOCKS);
	spmv_csr_scalar_kernel<int, float> <<<NUM_BLOCKS, BLOCK_SIZE>>> 
	    (numRows, csrRow, cooColIdx, cooVal, x, y, alpha, beta);
}

void blockMatMult(int *cooRowHostIdx, int *cooColHostIdx, float *cooValHost, float *xHost, float *yHost, int n, int nnz, 
		int *cooRowIdx, int *cooColIdx, float *cooVal, float *x, float *y, int *csrRow,
		int xOffset, int nPerTurn){
	clock_t tt;
	cudaError_t cudaStat;
	printf("starting memcpy to device\n");
	tt = clock();
	//FIXME: it's not necessary for CSR!
	//cudaStat = cudaMemcpy(cooRowIdx, cooRowHostIdx, nnz * sizeof(int), cudaMemcpyHostToDevice);
	//handleError(cudaStat);
	cudaStat = cudaMemcpy(cooColIdx, cooColHostIdx, nnz * sizeof(int), cudaMemcpyHostToDevice);
	handleError(cudaStat);
	cudaStat = cudaMemcpy(cooVal, cooValHost, nnz * sizeof(float), cudaMemcpyHostToDevice);
	handleError(cudaStat);
	cudaStat = cudaMemcpy(x, xHost, n * sizeof(float), cudaMemcpyHostToDevice);
	handleError(cudaStat);
	cudaDeviceSynchronize();
	reportTimeRound("memcpy", tt);
	reportTime(tt0);
	printf("starting multiplication\n");
	tt = clock();

	spmv_csr_scalar(nPerTurn, csrRow, cooColIdx, cooVal, x, y, DAMPINGFACTOR, (1-DAMPINGFACTOR)/n);
	cudaDeviceSynchronize();

	reportTimeRound("multiplication",tt);
	reportTime(tt0);

	cudaStat = cudaMemcpy(yHost + xOffset , y, nPerTurn * sizeof(float), cudaMemcpyDeviceToHost);
	handleError(cudaStat);

}

int main(){
	int	*cooRowHostIdx, *cooColHostIdx;
	float	*cooValHost;
	float	*xHost, *yHost;
	clock_t tt;

	//cudaThreadExit();
	//cudaSetDevice(1);

	tt0 = clock();
	time(&realt0);
	int MAX_BLOCKS = max_active_blocks(spmv_csr_scalar_kernel<int, float>, 512, (size_t) 0);
	printf("%d\n", MAX_BLOCKS);
	cooRowHostIdx = (int *) malloc(nnz * sizeof(int));
	cooColHostIdx = (int *) malloc(nnz * sizeof(int));
	cooValHost = (float *) malloc(nnz * sizeof(float));

	readBinMatrix(cooRowHostIdx, cooColHostIdx, cooValHost, nnz);
	//readMatrix(cooRowHostIdx, cooColHostIdx, cooValHost, nnz);
	for(int i=0;i<10;i++){
		printf("%d\t%d\t%.9f\n", cooRowHostIdx[i], cooColHostIdx[i], cooValHost[i]);
	}
	//readSampleMatrix(cooRowHostIdx, cooColHostIdx, cooValHost, nnz);
	xHost = (float *) malloc(n * sizeof(float));
	yHost = (float *) malloc(n * sizeof(float));
	for(int i=0;i<n;i++)xHost[i] = 1.0/n;
	
	int	*cooRowIdx, *cooColIdx;
	float	*cooVal;
	int	*csrRow;
	float	*x, *y;
	cudaError_t cudaStat;

	const unsigned int maxNNZPerTurn = min(500000000,nnz);
	const unsigned int maxNPerTurn = min(maxNNZPerTurn, n);

	//cudaStat = cudaMalloc((void **)&cooRowIdx, maxNNZPerTurn * sizeof(int));
	//handleError(cudaStat);
	//FIXME: cooRowIdx is not needed for CSR.
	cooRowIdx = NULL;
	cudaStat = cudaMalloc((void **)&cooColIdx, maxNNZPerTurn * sizeof(int));
	handleError(cudaStat);
	cudaStat = cudaMalloc((void **)&cooVal, maxNNZPerTurn * sizeof(float));
	handleError(cudaStat);
	cudaStat = cudaMalloc((void **)&x, n * sizeof(float));
	handleError(cudaStat);
	cudaStat = cudaMalloc((void **)&y, maxNPerTurn * sizeof(float));
	handleError(cudaStat);
	cudaStat = cudaMalloc((void**)&csrRow, (maxNPerTurn+1)*sizeof(int));
	handleError(cudaStat);
	reportTime(tt0);
	reportTimeReal();

	int *csrRowHost = (int *) malloc(sizeof(int) * (maxNPerTurn + 1));

	// starting block operation
	// for now we group edges
	unsigned int nnzCurTurn = maxNNZPerTurn;
	for(int iter = 0;iter<niter;iter++){
		printf("---------------\n");
		printf("iteration %d starting:\n", iter);
		clock_t t_iter = clock();
		int lastRow = -1;
		memset(yHost, 0, sizeof(float)*n);
		for(unsigned int cooOffset = 0, i=0; cooOffset<nnz; i++, cooOffset += maxNNZPerTurn){
			//convert to CSR
			printf("starting block operation\n");
			tt = clock();
			int j=0;
			int nCurTurn = 0;
			int xOffset = cooRowHostIdx[cooOffset];
			for(;j<maxNNZPerTurn&&j+cooOffset<nnz;nCurTurn++){
				csrRowHost[nCurTurn]=j;
				assert(nCurTurn <= maxNPerTurn);
				for(;j+cooOffset < nnz && cooRowHostIdx[cooOffset + j] <= nCurTurn + xOffset && j< maxNNZPerTurn;j++);
			}
			csrRowHost[nCurTurn] = j;
			nnzCurTurn = j;
			printf("%lld\n",(nCurTurn + 1) * sizeof(int));
			cudaStat = cudaMemcpy(csrRow, csrRowHost, sizeof(int) * (nCurTurn + 1), cudaMemcpyHostToDevice);
			handleError(cudaStat);
			printf("converting to CSR done.\n");

			float partialSum = 0;
			if(lastRow == cooRowHostIdx[cooOffset])
				partialSum = yHost[lastRow];
			assert(nCurTurn + xOffset <= n+1);

			blockMatMult(cooRowHostIdx+cooOffset, cooColHostIdx+cooOffset, cooValHost+cooOffset, xHost, yHost, n, nnzCurTurn,
					cooRowIdx, cooColIdx, cooVal, x, y, csrRow,
					xOffset, nCurTurn);

			yHost[lastRow] += partialSum;
			lastRow = xOffset + nCurTurn - 1;
			reportTimeRound("turn(block)",tt);
			reportTime(tt0);
			reportTimeReal();
		}
		memcpy(xHost, yHost, n*sizeof(float));
		reportTimeRound("iteration",t_iter);
		reportTime(tt0);
		reportTimeReal();
	}
	dumpRes(xHost);

	free(xHost);
	free(yHost);
	free(cooColHostIdx);
	free(cooRowHostIdx);
	free(cooValHost);
	free(csrRowHost);
	
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

