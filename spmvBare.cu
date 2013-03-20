#include<cstdio>
#include<cstdlib>
#include<ctime>
#include<cuda_runtime.h>
#include<cassert>
#include "common.h"


int main(){
	int	*cooRowHostIdx, *cooColHostIdx;
	float	*cooValHost;
	float	*xHost, *yHost;

	tt0 = clock();
	time(&realt0);
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
	
	for(int iter = 0;iter<niter;iter++){
		printf("---------------\n");
		printf("iteration %d starting:\n", iter);
		clock_t t_iter = clock();
		for(int i=0;i<nnz;i++){
			yHost[cooRowHostIdx[i]] += xHost[cooColHostIdx[i]] * cooValHost[i];
		}
		for(int i=12;i<n;i++){
			yHost[i] = (yHost[i] * DAMPINGFACTOR + (1 - DAMPINGFACTOR)/n);
		}
		memcpy(xHost, yHost, n * sizeof(float));
		reportTimeReal();
	}
	dumpRes(xHost);

	free(xHost);
	free(yHost);
	free(cooColHostIdx);
	free(cooRowHostIdx);
	free(cooValHost);
	
	return 0;
}

