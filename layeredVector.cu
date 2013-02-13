#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <utility>
#define RANDRESET 0.15
typedef std::vector<std::pair<int, int> > vii;

//const int N=61578171;
const int N=6;

struct sum_functor{
	template <typename Tuple>
	__host__ __device__ 
		/*float operator()(const thrust::device_vector<float> & x, const int &deg){
			//FIXME: i don't know how to fix this with layered vector
			return thrust::reduce(x.begin(), x.end(), 0, thrust::plus<float>())/(float)deg;
		}*/
		//float operator()(const float *&x, const int &deg){
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
	thrust::host_vector<float *> hGroupedData(N);
	thrust::host_vector<int> hInDegree(N);
	thrust::host_vector<int> hOutDegree(N);
	vii invData;
	std::map<int,int> outDegreeTemp;
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
	sort(invData.begin(), invData.end());
	//TODO: also sort by #inDegrees
	int n = invData.size();
	for(int i=0;i<n;i++){
		int v=invData[i].first,u=invData[i].second;
		std::cout<<u<<"\t"<<v<<std::endl;
	}
	std::cout<<std::endl;
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
	int m = hGroupedData.size();
	//thrust::device_vector<float> dP(m,0);
	float *dP,*hP;
	const int sz = m*sizeof(float);
	cudaMalloc((void **)&dP, sz);
	thrust::device_vector<float *> dGroupedData(hGroupedData);
	thrust::device_vector<int> dInDegree(hInDegree);
	thrust::device_vector<int> dOutDegree(hOutDegree);
	thrust::constant_iterator<float *> dPfirst(dP);
	thrust::constant_iterator<float *> dPlast = dPfirst + m;
	thrust::device_ptr<float> dPptr(dP);
	hP = (float *)malloc(sz);
	thrust::fill(dPptr, dPptr+m, 1);
	std::cout<<"here"<<std::endl;
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(dGroupedData.begin(),dInDegree.begin(),dOutDegree.begin(),dPfirst,dPptr)),
			 thrust::make_zip_iterator(thrust::make_tuple(dGroupedData.end(), dInDegree.end(), dOutDegree.end(), dPlast, dPptr + m)),
			 sum_functor());
	//thrust::copy(dP.begin(), dP.end(), hP.begin());
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
	return 0;
}
