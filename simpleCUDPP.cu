// NOTE: when nvcc is updated, cudpp has to be recompiled.
// Compile with -lcudpp
#include<cstring>
#include<cstdio>
#include<string>
#include<cuda.h>
#include<cudpp.h>
void computeSumScanGold(float *r, float *i, int n, CUDPPConfiguration c){
	r[0]=0;
	for(int j=1;j<=n;j++){
		r[j]=r[j-1]+i[j-1];
	}
}
int main( int argc, char** argv) 
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "error: no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }
    int dev = 0;
    if (argc > 1) {
        std::string arg = argv[1];
        size_t pos = arg.find("=");
        if (arg.find("device") && pos != std::string::npos) {
            dev = atoi(arg.c_str() + (pos + 1));
        }
    }
    if (dev < 0) dev = 0;
    if (dev > deviceCount-1) dev = deviceCount - 1;
    cudaSetDevice(dev);

    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess)
    {
        printf("Using device %d:\n", dev);
        printf("%s; global mem: %dB; compute v%d.%d; clock: %d kHz\n",
               prop.name, (int)prop.totalGlobalMem, (int)prop.major, 
               (int)prop.minor, (int)prop.clockRate);
    }
    unsigned int numElements = 32168;
    unsigned int memSize = sizeof( float) * numElements;

    // allocate host memory
    float* h_idata = (float*) malloc( memSize);
    // initalize the memory
    for (unsigned int i = 0; i < numElements; ++i) 
    {
        h_idata[i] = (float) (rand() & 0xf);
    }

    // allocate device memory
    float* d_idata;
    cudaError_t result = cudaMalloc( (void**) &d_idata, memSize);
    printf("Allocate device memory done.\n");
    if (result != cudaSuccess) {
        printf("ErrorAlloc: %s\n", cudaGetErrorString(result));
        exit(-1);
    }
    
    // copy host memory to device
    result = cudaMemcpy( d_idata, h_idata, memSize, cudaMemcpyHostToDevice);
    if (result != cudaSuccess) {
        printf("ErrorCopy: %s\n", cudaGetErrorString(result));
        exit(-1);
    }
     
    // allocate device memory for result
    float* d_odata;
    result = cudaMalloc( (void**) &d_odata, memSize);
    printf("Allocate device memory for output done.\n");
    if (result != cudaSuccess) {
        printf("ErrorMalloc: %s\n", cudaGetErrorString(result));
        exit(-1);
    }

    // Initialize the CUDPP Library
    CUDPPHandle theCudpp;
    cudppCreate(&theCudpp);

    CUDPPConfiguration config;
    config.op = CUDPP_ADD;
    config.datatype = CUDPP_FLOAT;
    config.algorithm = CUDPP_SCAN;
    config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;
    
    CUDPPHandle scanplan = 0;
    CUDPPResult res = cudppPlan(theCudpp, &scanplan, config, numElements, 1, 0);  

    if (CUDPP_SUCCESS != res)
    {
        printf("Error creating CUDPPPlan\n");
        exit(-1);
    }

    printf("ready for scan.\n");
    // Run the scan
    res = cudppScan(scanplan, d_odata, d_idata, numElements);
    if (CUDPP_SUCCESS != res)
    {
        printf("Error in cudppScan()\n");
        exit(-1);
    }

    printf("done for scan.\n");
    // allocate mem for the result on host side
    float* h_odata = (float*) malloc( memSize);
    // copy result from device to host
    result = cudaMemcpy( h_odata, d_odata, memSize, cudaMemcpyDeviceToHost);
    if (result != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(result));
        exit(-1);
    }
    
    // compute reference solution
    float* reference = (float*) malloc( memSize);
    computeSumScanGold( reference, h_idata, numElements, config);

    // check result
    bool passed = true;
    for (unsigned int i = 0; i < numElements; i++)
        if (reference[i] != h_odata[i]) passed = false;

    for(int i=0;i<10; i++){
	    printf("%f %f\n",reference[i], h_odata[i]);
    }
        
    printf( "Test %s\n", passed ? "PASSED" : "FAILED");

    res = cudppDestroyPlan(scanplan);
    if (CUDPP_SUCCESS != res)
    {
        printf("Error destroying CUDPPPlan\n");
        exit(-1);
    }

    // shut down the CUDPP library
    cudppDestroy(theCudpp);
    
    free( h_idata);
    free( h_odata);
    free( reference);
    cudaFree(d_idata);
    cudaFree(d_odata);
    return 0;
}
