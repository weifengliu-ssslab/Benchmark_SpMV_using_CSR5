#compilers
CC=c++

#GLOBAL_PARAMETERS
VALUE_TYPE = double
NUM_RUN = 1000

#ENVIRONMENT_PARAMETERS
#OPRIONS=-xCORE-AVX2 -opt-prefetch=3 #icc/icpc option
OPRIONS=-mavx -mavx2 -march=native #c++ option
#backup
#$(CC) -xCORE-AVX2 -opt-prefetch=3 -Wno-deprecated-writable-strings -fopenmp -O3 main.cpp -o spmv -D VALUE_TYPE=$(VALUE_TYPE) -D NUM_RUN=$(NUM_RUN)
avx2:
	$(CC) $(OPRIONS) -fopenmp -O3 main.cpp -o spmv -D VALUE_TYPE=$(VALUE_TYPE) -D NUM_RUN=$(NUM_RUN)
