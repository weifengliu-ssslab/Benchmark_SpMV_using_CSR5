#compilers
CC=icc

#GLOBAL_PARAMETERS
VALUE_TYPE = double
NUM_RUN = 1000

#ENVIRONMENT_PARAMETERS
#source /opt/intel/composer_xe_2015.1.133/bin/compilervars.sh intel64
#export OMP_NUM_THREADS=236;export KMP_AFFINITY=default

phi:
	$(CC) -O2 -ansi-alias -opt-streaming-cache-evict=3 -opt-prefetch=3 -opt-prefetch-distance=64,8 -opt-streaming-stores always -openmp main.cpp -o spmv -D VALUE_TYPE=$(VALUE_TYPE) -D NUM_RUN=$(NUM_RUN)
