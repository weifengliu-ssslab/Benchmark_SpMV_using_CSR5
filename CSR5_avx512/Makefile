#compilers
CC=icc

#GLOBAL_PARAMETERS
VALUE_TYPE = double
NUM_RUN = 1000

#ENVIRONMENT_PARAMETERS
#source /opt/intel/composer_xe_2015.1.133/bin/compilervars.sh intel64
#export OMP_NUM_THREADS=236;export KMP_AFFINITY=default

phi:
	$(CC) -O3 -xMIC-AVX512 -qopenmp main.cpp -o spmv -D VALUE_TYPE=$(VALUE_TYPE) -D NUM_RUN=$(NUM_RUN)
