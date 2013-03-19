//FIXME: what if n is too large for one turn?
#include<cstdio>
#include<cstdlib>
#include<ctime>
#include<cuda_runtime.h>
#include<cassert>
#include<sstream>
#include "common.h"

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
	int T_BLOCKS = (int)DIVIDE_INTO(numRows, BLOCK_SIZE);
	dim3 block(BLOCK_SIZE);
	dim3 grid;
	grid.x = T_BLOCKS % 65535;
	grid.y = (T_BLOCKS / 65535 + 1);
	spmv_csr_scalar_kernel<int, float> <<<grid, block>>> 
	    (numRows, csrRow, cooColIdx, cooVal, x, y, alpha, beta);
}

unsigned int matCoo2Csr(int *col, int *row, int *csr, int m){
	unsigned int nCurTurn = 0;
	int j=0;
	for(; j < m; nCurTurn++){
		csr[nCurTurn] = j;
		for(;j < m && row[j] <= nCurTurn; j++);
	}
	csr[nCurTurn] = j;
	return nCurTurn;
}

unsigned int loadBlockMatrixCoo(int *col, int *row, float *val, int shard){
	std::string filename (mtxBinFile);
	std::stringstream basefile(filename);
	basefile<<"."<<shard;
	filename = basefile.str();
	std::string colfile = filename + ".col";
	std::string rowfile = filename + ".row";
	std::string valfile = filename + ".val";
	std::string metafile = filename + ".meta";
	unsigned int m;
	clock_t tt = clock();
	FILE *fp = fopen(metafile.c_str(), "r");
	fscanf(fp, "%d", &m);
	fclose(fp);
	FILE *fprow = fopen(rowfile.c_str(),"rb");
	FILE *fpcol = fopen(colfile.c_str(),"rb");
	FILE *fpval = fopen(valfile.c_str(),"rb");
	fread(row, sizeof(int), m, fprow);
	fread(col, sizeof(int), m, fpcol);
	fread(val, sizeof(float), m, fpval);
	fclose(fprow);
	fclose(fpcol);
	fclose(fpval);
	printf("Read matrix in %.3fs\n", ((double)clock() - tt)/CLOCKS_PER_SEC);
	return m;
}

int main(){
	int	*cooRowHostIdx, *cooColHostIdx;
	float	*cooValHost;
	float	*xHost, *yHost;

	tt0 = clock();
	time(&realt0);
	
	cooRowHostIdx = (int *) malloc(nnz * sizeof(int));
	cooColHostIdx = (int *) malloc(nnz * sizeof(int));
	cooValHost = (float *) malloc(nnz * sizeof(float));

	readMetaMatrix();

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

