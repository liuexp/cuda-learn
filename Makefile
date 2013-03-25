spmv:
	nvcc -arch=sm_20 -O2 spmv.cu

spmv2:
	nvcc -I ./multicopy/common/inc/ -arch=sm_20 -Xcompiler -Wall spmv2.cu

all: spmv
