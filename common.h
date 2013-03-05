
#if THRUST_VERSION >= 100700
#include <thrust/system/cuda/detail/detail/launch_calculator.h>
#elif THRUST_VERSION >= 100600
#include <thrust/system/cuda/detail/arch.h>
#else
#include <thrust/detail/backend/cuda/arch.h>
#endif

#include<map>

//const float RANDRESET = 0.15;
const float DAMPINGFACTOR = 0.85;
const char mtxFile[] = "/media/tmp/graphchi/data/test3";
const char mtxBinFile[] = "/media/tmp/graphchi/data/test4";
const char mtxBinRowFile[] = "/media/tmp/graphchi/data/test4row";
const char mtxBinColFile[] = "/media/tmp/graphchi/data/test4col";
const char mtxBinValFile[] = "/media/tmp/graphchi/data/test4val";
const char resFile[] = "res2";
const unsigned int n = 61578414;
const unsigned int nnz =1446476275;
//const int n = 4, nnz = 9;
const int niter = 4;
const int topK = 20;
clock_t tt0;

typedef struct adjTuple{
	int u,v;
	float *val;
} adjTuple;

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
			//printf("success memcpy\n");
			break;
		default:
			printf("unknown error:%s\n", cudaGetErrorString(z));
			exit(EXIT_FAILURE);
			break;
	}

}
void FIXLINE(char *s){
	int l = (int)strlen(s) - 1;
	if(s[l] == '\n')s[l] = 0;
}

void reportTime(clock_t tt0){
	printf("-- ELAPSED TIME: %.3fs\n", ((double)clock() - tt0)/CLOCKS_PER_SEC);
}

void reportTimeRound(const char *s, clock_t tt){
	printf("%s done in %.3fs\n", s, ((double)clock() - tt)/CLOCKS_PER_SEC);
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

void readBinMatrix(int *row, int *col, float *val, int m){
	clock_t tt = clock();
	FILE *fprow = fopen(mtxBinRowFile,"rb");
	FILE *fpcol = fopen(mtxBinColFile,"rb");
	FILE *fpval = fopen(mtxBinValFile,"rb");
	fread(row, sizeof(int), m, fprow);
	fread(col, sizeof(int), m, fpcol);
	fread(val, sizeof(float), m, fpval);
	fclose(fprow);
	fclose(fpcol);
	fclose(fpval);
	printf("Read matrix in %.3fs\n", ((double)clock() - tt)/CLOCKS_PER_SEC);
}

void readMatrix(int *row, int *col, float *val, int m){
	FILE *fp = fopen(mtxBinFile,"r");
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


void dumpRes(float *xHost){
	for(int i=0;i<min(n,30);i++){
		printf("%d\t%.10f\n", i, xHost[i]);
	}
//	FILE *fres = fopen(resFile, "w");
//	for(int i=0;i<n;i++){
//		fprintf(fres, "%.10f\n", xHost[i]);
//	}
//	fclose(fres);
	
	//choose only topK
	std::map<float, int> tmp;
	for(int i=0;i<n;i++){
		tmp[xHost[i]] = i;
		if(tmp.size() > topK)
			tmp.erase(tmp.begin());
	}
	for(std::map<float, int >::iterator itr=tmp.begin();itr!=tmp.end();itr++){
		printf("%d\t%.10f\n", itr->second, itr->first);
	}
}

template <typename KernelFunction>
size_t max_active_blocks(KernelFunction kernel, const size_t CTA_SIZE, const size_t dynamic_smem_bytes)
{
#if THRUST_VERSION >= 100700
  using namespace thrust::system::cuda::detail;
  function_attributes_t attributes = function_attributes(kernel);
  device_properties_t properties = device_properties();
  return properties.multiProcessorCount * cuda_launch_config_detail::max_active_blocks_per_multiprocessor(properties, attributes, CTA_SIZE, dynamic_smem_bytes);
#elif THRUST_VERSION >= 100600
  return thrust::system::cuda::detail::arch::max_active_blocks(kernel, CTA_SIZE, dynamic_smem_bytes);
#else
  return thrust::detail::backend::cuda::arch::max_active_blocks(kernel, CTA_SIZE, dynamic_smem_bytes);
#endif
}

template <typename Size1, typename Size2>
__host__ __device__ 
Size1 DIVIDE_INTO(Size1 N, Size2 granularity)
{
  return (N + (granularity - 1)) / granularity;
}


