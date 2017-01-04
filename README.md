# Benchmark_SpMV_using_CSR5

<br><hr>
<h3>Introduction</h3>

This is the source code of the paper

Weifeng Liu and Brian Vinter, "CSR5: An Efficient Storage Format for Cross-Platform Sparse Matrix-Vector Multiplication". In Proceedings of the 29th ACM international conference on Supercomputing (ICS '15), pp.339-350, 2015. 
[<a href="http://www.nbi.dk/~weifeng/papers/CSR5_Liu_ics15.pdf">pdf</a>][<a href="http://www.nbi.dk/~weifeng/slides/CSR5_Liu_ics15_slides.pptx">slides</a>]

Contact: <a href="http://www.nbi.dk/~weifeng/">Weifeng Liu</a> and Brian Vinter (vinter _at_ nbi.ku.dk).

Updates:  
&nbsp;&nbsp;&nbsp;(Jan 2017, avx512 and opencl): added two versions: AVX-512 for Knights Landing Phi (KNL) and OpenCL for nVidia GPUs.  
&nbsp;&nbsp;&nbsp;(Jul 2016, phi): fixed the same two issues in the original AVX2 version. Thanks Jan Philipp Ecker!  
&nbsp;&nbsp;&nbsp;(Jul 2016, avx2): Improved performance of y-vector update. Thanks Jan Philipp Ecker!  
&nbsp;&nbsp;&nbsp;(Jul 2016, avx2): Fixed a bug in processing small matrices. Thanks Jan Philipp Ecker!  
&nbsp;&nbsp;&nbsp;(Apr 2016, cuda): Fixed a bug in timing. Thanks Shigang Li!  
  
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
<h3>Intel Xeon Knights Landing Phi (KNL) AVX-512 version</h3>

- Execution

1. Set environments for the Intel C/C++ Compilers. For example, use ``source /opt/intel/composer_xe_2015.1.133/bin/compilervars.sh intel64``,
2. Run ``make``,
3. Run ``./spmv example.mtx``.

- Tested environments

1. Intel Xeon Kinghts Landing Phi (KNL) 7210 in a host with CentOS 7.2 64-bit Linux installed.

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
<h3>nVidia GPU (OpenCL) version</h3>

- Execution

1. Set OpenCL path in the Makefile,
2. Run ``make``,
3. Run ``./spmv example.mtx``.

- Tested environments

1. nVidia Pascal GTX 1060 GPU in a host with Ubuntu 15.04 64-bit Linux installed.

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

