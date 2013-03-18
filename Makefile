spmv:
	nvcc -arch=sm_20 -O2 spmv.cu

all: spmv
