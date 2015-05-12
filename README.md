# Benchmark_SpMV_using_CSR5

<br><hr>
<h3>Introduction</h3>

This is the source code of the paper

Weifeng Liu and Brian Vinter. 2015. CSR5: An Efficient Storage Format for Cross-Platform Sparse Matrix-Vector Multiplication. 
In Proceedings of the 29th ACM international conference on Supercomputing (ICS '15). (To appear).

The camera-ready PDF is downloadable at <a href="http://arxiv.org/abs/1503.05032">http://arxiv.org/abs/1503.05032</a>.

Contact: <a href="http://www.nbi.dk/~weifeng/">Weifeng Liu</a> and Brian Vinter (vinter _at_ nbi.ku.dk).

<br><hr>
<h3>CPU (AVX2) version</h3>

- Execution

1. Set environments for the Intel C/C++ Compilers. For example, use ``source /opt/intel/composer_xe_2015.1.133/bin/compilervars.sh intel64``,
2. Run ``make``,
3. Run ``./spmv example.mtx``.

- Tested environments

1. Intel Core i7-4770R CPU with Ubuntu 14.04 64-bit Linux installed.
2. Intel Xeon E5-2667 v3 dual-socket CPUs with Redhat 6.5 64-bit Linux installed.

- Data type

1. Currently, only 64-bit double precision SpMV is supported.

<br><hr>
<h3>nVidia GPU (CUDA) version</h3>

- Execution

1. Set CUDA path in the Makefile,
2. Run ``make``,
3. Run ``./spmv example.mtx``.

- Tested environments

1. nVidia GeForce GTX 980 GPU in a host with Ubuntu 14.04 64-bit Linux installed.
2. nVidia GeForce GT 650M GPU in a host with Mac OS X 10.9.2 installed.

- Data type

1. The code supports both double precision and single precision SpMV. Use ``make VALUE_TYPE=double`` for double precision or ``make VALUE_TYPE=float`` for single precision.

<br><hr>
<h3>AMD GPU (OpenCL) version</h3>

- Execution

1. Set OpenCL path in the Makefile,
2. Run ``make``,
3. Run ``./spmv example.mtx``.

- Tested environments

1. AMD Radeon R9-290X GPU in a host with Ubuntu 14.04 64-bit Linux installed.

- Data type

1. The code supports both double precision and single precision SpMV. Use ``make VALUE_TYPE=double`` for double precision or ``make VALUE_TYPE=float`` for single precision.

<br><hr>
<h3>Intel Xeon Phi (KNC) version</h3>

- Execution

1. Set environments for the Intel C/C++ Compilers. For example, use ``source /opt/intel/composer_xe_2015.1.133/bin/compilervars.sh intel64``,
2. Run ``make``,
3. Run ``./spmv example.mtx``.

- Tested environments

1. Intel Xeon Phi 5110p in a host with Redhat 6.5 64-bit Linux installed.

- Data type

1. Currently, only 64-bit double precision SpMV is supported.
