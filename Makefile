spmv:
	nvcc -arch=sm_20 -O2 spmv.cu

spmv2:
	nvcc -arch=sm_20 -O2 spmv2.cu

all: spmv
