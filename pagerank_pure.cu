#include <cstdio>
#include <cstdlib>
#include <vector>
#include <map>
#include <utility>
#include <algorithm>
#define RANDRESET 0.15
typedef std::vector<std::pair<int, int> > vii;

//const int N=61578171;
const int N=6;

struct sum_functor{
	template <typename Tuple>
	__host__ __device__ 
		void operator()(Tuple t){
			thrust::device_ptr<float> P = thrust::device_pointer_cast(thrust::get<3>(t));
			const int &outDeg = thrust::get<2>(t);
			const int &inDeg = thrust::get<1>(t);
			//float *x = thrust::get<0>(t);
			thrust::device_ptr<float> x = thrust::device_pointer_cast(thrust::get<0>(t));
			float ret = 0;
			for(int i=0;i<inDeg;i++){
				//ret += P[x[i]];
				ret += *(P + *(x+i));
			}
			thrust::get<4>(t) = (RANDRESET + (1-RANDRESET)*ret);
		}
};

void FIXLINE(char *s){
	int l = (int)strlen(s)-1;
	if(s[l] == '\n')s[l]=0;
}

int main(){
	float ** hGroupedData;
	int *hInDegree,*hOutDegree;
	vii invData;
	std::map<int,int> outDegreeTemp;
	int sz = N * sizeof(float*);
	hGroupedData = (float **)malloc(sz);
	sz = N*sizeof(int);
	hInDegree = (int *)malloc(sz);
	hOutDegree = (int *)malloc(sz);
	FILE *fp = fopen("testdata","r");
	char s[1024];
	while(fgets(s, 1024, fp) != NULL){
		FIXLINE(s);
		char del[] = "\t ";
		if(s[0]=='#' || s[0] == '%') continue;
		char *t;
		int a,b;
		t=strtok(s,del);
		a=atoi(t);
		t=strtok(NULL,del);
		b=atoi(t);
		invData.push_back(std::make_pair(b,a));
	}
	std::sort(invData.begin(), invData.end());
	//TODO: also sort by #inDegrees
	int n = invData.size();
	for(int i=0;i<n;i++){
		int v=invData[i].first,u=invData[i].second;
		if(outDegreeTemp.find(u)==outDegreeTemp.end())
			outDegreeTemp[u]=1;
		else outDegreeTemp[u]=outDegreeTemp[u]+1;
	}
	int startv,cntIn=0;
	startv=-1;
	for(int i=0;i<n;i++){
		int v=invData[i].first,u=invData[i].second;
		if(v != startv){
			//new vertex
			if(cntIn != 0){
				float *dtmp,*htmp;
				const int sz = cntIn * sizeof(float);
				cudaMalloc((void **)&dtmp, sz);
				htmp = (float *)malloc(sz);
				for(int j=startv;j<i;j++){
					htmp[j-startv]=invData[j].second;
				}
				cudaMemcpy(dtmp, htmp, sz, cudaMemcpyHostToDevice);
				free(htmp);
				hGroupedData[startv]=dtmp;
				hInDegree[startv]=cntIn;
				hOutDegree[startv]=outDegreeTemp[startv];
			}

			startv = i;
			cntIn = 0;
		}
		cntIn++;
	}
	int m = N;
	//thrust::device_vector<float> dP(m,0);
	float *dP,*hP;
	sz = m*sizeof(float);
	cudaMalloc((void **)&dP, sz);
	float **dGroupedData;
	int *dInDegree,*dOutDegree;
	sz = m*sizeof(float**);
	cudaMalloc((void **)&dGroupedData,sz);
	sz = m*sizeof(int*);
	cudaMalloc((void**)&dInDegree, sz);
	cudaMalloc((void**)&dOutDegree, sz);
	hP = (float *)malloc(sz);
	for(int i=0;i<N;i++)hP[i]=1;
	cudaMemcpy(dP, hP, sz, cudaMemcpyHostToDevice);
	//thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(dGroupedData.begin(),dInDegree.begin(),dOutDegree.begin(),dPfirst,dPptr)),
	//		 thrust::make_zip_iterator(thrust::make_tuple(dGroupedData.end(), dInDegree.end(), dOutDegree.end(), dPlast, dPptr + m)),
	//		 sum_functor());
	//thrust::copy(dP.begin(), dP.end(), hP.begin());
	//FIXME:which way to do memory alignment here is simpler?
	cudaMemcpy(hP, dP, sz, cudaMemcpyDeviceToHost);
	cudaFree(dP);
	for(int i=0;i<m;i++){
		float *tmp = hGroupedData[i];
		cudaFree(tmp);
	}
	for(int i=0;i<m;i++){
		printf("%d:\t%.4f\n",i,(float)hP[i]/hOutDegree[i]);
	}
	fclose(fp);
	free(hP);
	free(hOutDegree);
	free(hInDegree);
	free(hGroupedData);
	return 0;
}
