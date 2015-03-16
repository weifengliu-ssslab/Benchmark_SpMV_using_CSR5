#compilers
CC=g++

#GLOBAL_PARAMETERS
VALUE_TYPE = double
NUM_RUN = 1000

#ENVIRONMENT_PARAMETERS


#OPENCL_PARAMETERS
OPENCL_LIBS = -L/usr/lib -lamdocl64

opencl:
	$(CC) -Wno-write-strings -Wno-unused-result -O3 main.cpp -o spmv $(OPENCL_LIBS) -D VALUE_TYPE=$(VALUE_TYPE) -D NUM_RUN=$(NUM_RUN)
