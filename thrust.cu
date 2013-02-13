#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>

#include <iostream>

int main(){
	const int N = 108000000;
	const int M =  70000000;
	thrust :: device_vector <int> D(N, 0);
	thrust :: fill(D.begin(), D.end(), 1);
	for(int i=0;i<10;i++){
		std::cout<<"D["<<i<<"]="<<D[i]<<std::endl;
	}
	thrust::host_vector <int> H(M);
	for(int i=0;i<1000;i++){
		thrust::sequence(H.begin(), H.begin()+10);
		thrust::copy(H.begin(), H.end(), D.begin());
	}
	for(int i=0;i<10;i++){
		std::cout<<"D["<<i<<"]="<<D[i]<<std::endl;
	}
	return 0;
}
