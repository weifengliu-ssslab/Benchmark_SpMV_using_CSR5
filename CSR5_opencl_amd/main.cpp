#include <iostream>

#include "anonymouslib_opencl.h"

#include "mmio.h"

using namespace std;

#ifndef VALUE_TYPE
#define VALUE_TYPE double
#endif

#ifndef NUM_RUN
#define NUM_RUN 1000
#endif

int call_anonymouslib(int m, int n, int nnzA,
                  int *csrRowPtrA, int *csrColIdxA, VALUE_TYPE *csrValA,
                  VALUE_TYPE *x, VALUE_TYPE *y, VALUE_TYPE alpha)
{
    int err = 0;

    // set device
    BasicCL basicCL;

    char platformVendor[CL_STRING_LENGTH];
    char platformVersion[CL_STRING_LENGTH];

    char gpuDeviceName[CL_STRING_LENGTH];
    char gpuDeviceVersion[CL_STRING_LENGTH];
    int  gpuDeviceComputeUnits;
    cl_ulong  gpuDeviceGlobalMem;
    cl_ulong  gpuDeviceLocalMem;

    cl_uint             numPlatforms;           // OpenCL platform
    cl_platform_id*     cpPlatforms;

    cl_uint             numGpuDevices;          // OpenCL Gpu device
    cl_device_id*       cdGpuDevices;

    cl_context          cxGpuContext;           // OpenCL Gpu context
    cl_command_queue    cqGpuCommandQueue;      // OpenCL Gpu command queues

    bool profiling = true;
    int select_device = 0;

    // platform
    err = basicCL.getNumPlatform(&numPlatforms);
    if(err != CL_SUCCESS) return err;
    cout << "platform number: " << numPlatforms << ".  ";

    cpPlatforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * numPlatforms);

    err = basicCL.getPlatformIDs(cpPlatforms, numPlatforms);
    if(err != CL_SUCCESS) return err;

    for (unsigned int i = 0; i < numPlatforms; i++)
    {
        err = basicCL.getPlatformInfo(cpPlatforms[i], platformVendor, platformVersion);
        if(err != CL_SUCCESS) return err;

        // Gpu device
        err = basicCL.getNumGpuDevices(cpPlatforms[i], &numGpuDevices);

        if (numGpuDevices > 0)
        {
            cdGpuDevices = (cl_device_id *)malloc(numGpuDevices * sizeof(cl_device_id) );

            err |= basicCL.getGpuDeviceIDs(cpPlatforms[i], numGpuDevices, cdGpuDevices);

            err |= basicCL.getDeviceInfo(cdGpuDevices[select_device], gpuDeviceName, gpuDeviceVersion,
                                         &gpuDeviceComputeUnits, &gpuDeviceGlobalMem,
                                         &gpuDeviceLocalMem, NULL);
            if(err != CL_SUCCESS) return err;

            cout << "Platform [" << i <<  "] Vendor: " << platformVendor << ", Version: " << platformVersion << endl;
            cout << "Using GPU device: " //<< numGpuDevices << " Gpu device: "
                 << gpuDeviceName << " ("
                 << gpuDeviceComputeUnits << " CUs, "
                 << gpuDeviceLocalMem / 1024 << " kB local, "
                 << gpuDeviceGlobalMem / (1024 * 1024) << " MB global, "
                 << gpuDeviceVersion << ")" << endl;

            break;
        }
        else
        {
            continue;
        }
    }

    // Gpu context
    err = basicCL.getContext(&cxGpuContext, cdGpuDevices, numGpuDevices);
    if(err != CL_SUCCESS) return err;

    // Gpu commandqueue
    if (profiling)
        err = basicCL.getCommandQueueProfilingEnable(&cqGpuCommandQueue, cxGpuContext, cdGpuDevices[select_device]);
    else
        err = basicCL.getCommandQueue(&cqGpuCommandQueue, cxGpuContext, cdGpuDevices[select_device]);
    if(err != CL_SUCCESS) return err;







    double gb = getB<int, VALUE_TYPE>(m, nnzA);
    double gflop = getFLOP<int>(nnzA);

    // Define pointers of matrix A, vector x and y
    cl_mem      d_csrRowPtrA;
    cl_mem      d_csrColIdxA;
    cl_mem      d_csrValA;
    cl_mem      d_x;
    cl_mem      d_y;
    cl_mem      d_y_bench;

    // Matrix A
    d_csrRowPtrA = clCreateBuffer(cxGpuContext, CL_MEM_READ_ONLY, (m+1) * sizeof(int), NULL, &err);
    if(err != CL_SUCCESS) return err;
    d_csrColIdxA = clCreateBuffer(cxGpuContext, CL_MEM_READ_ONLY, nnzA  * sizeof(int), NULL, &err);
    if(err != CL_SUCCESS) return err;
    d_csrValA    = clCreateBuffer(cxGpuContext, CL_MEM_READ_ONLY, nnzA  * sizeof(VALUE_TYPE), NULL, &err);
    if(err != CL_SUCCESS) return err;

    err = clEnqueueWriteBuffer(cqGpuCommandQueue, d_csrRowPtrA, CL_TRUE, 0, (m+1) * sizeof(int), csrRowPtrA, 0, NULL, NULL);
    if(err != CL_SUCCESS) return err;
    err = clEnqueueWriteBuffer(cqGpuCommandQueue, d_csrColIdxA, CL_TRUE, 0, nnzA  * sizeof(int), csrColIdxA, 0, NULL, NULL);
    if(err != CL_SUCCESS) return err;
    err = clEnqueueWriteBuffer(cqGpuCommandQueue, d_csrValA, CL_TRUE, 0, nnzA  * sizeof(VALUE_TYPE), csrValA, 0, NULL, NULL);
    if(err != CL_SUCCESS) return err;

    // Vector x
    d_x    = clCreateBuffer(cxGpuContext, CL_MEM_READ_ONLY, n  * sizeof(VALUE_TYPE), NULL, &err);
    if(err != CL_SUCCESS) return err;
    err = clEnqueueWriteBuffer(cqGpuCommandQueue, d_x, CL_TRUE, 0, n  * sizeof(VALUE_TYPE), x, 0, NULL, NULL);
    if(err != CL_SUCCESS) return err;

    // Vector y
    d_y    = clCreateBuffer(cxGpuContext, CL_MEM_READ_WRITE, m  * sizeof(VALUE_TYPE), NULL, &err);
    if(err != CL_SUCCESS) return err;
    memset(y, 0, m  * sizeof(VALUE_TYPE));
    err = clEnqueueWriteBuffer(cqGpuCommandQueue, d_y, CL_TRUE, 0, m  * sizeof(VALUE_TYPE), y, 0, NULL, NULL);
    if(err != CL_SUCCESS) return err;

    d_y_bench    = clCreateBuffer(cxGpuContext, CL_MEM_READ_WRITE, m  * sizeof(VALUE_TYPE), NULL, &err);
    if(err != CL_SUCCESS) return err;
    err = clEnqueueWriteBuffer(cqGpuCommandQueue, d_y_bench, CL_TRUE, 0, m  * sizeof(VALUE_TYPE), y, 0, NULL, NULL);
    if(err != CL_SUCCESS) return err;




    double time = 0;

    anonymouslibHandle<int, unsigned int, VALUE_TYPE> A(m, n);
    err = A.setOCLENV(cxGpuContext, cqGpuCommandQueue);
    //cout << "setOCLENV err = " << err << endl;

    err = A.inputCSR(nnzA, d_csrRowPtrA, d_csrColIdxA, d_csrValA);
    //cout << "inputCSR err = " << err << endl;

    err = A.setX(d_x); // you only need to do it once!
    //cout << "setX err = " << err << endl;

    err = A.setSigma(ANONYMOUSLIB_AUTO_TUNED_SIGMA);
    //cout << "setSigma err = " << err << endl;

    // warmup device
    A.warmup();
    err = clFinish(cqGpuCommandQueue);

    anonymouslib_timer asCSR5_timer;
    asCSR5_timer.start();

    err = A.asCSR5();
    err = clFinish(cqGpuCommandQueue);

    cout << "CSR->CSR5 time = " << asCSR5_timer.stop() << " ms." << endl;
    //cout << "asCSR5 err = " << err << endl;

    // check correctness by running 1 time
    err = A.spmv(alpha, d_y, &time);
    //cout << "spmv err = " << err << endl;
    err = clEnqueueReadBuffer(cqGpuCommandQueue, d_y, CL_TRUE, 0, m * sizeof(VALUE_TYPE), y, 0, NULL, NULL);
    if(err != CL_SUCCESS) return err;

    // warm up by running 50 times
    if (NUM_RUN)
    {
        for (int i = 0; i < 50; i++)
            err = A.spmv(alpha, d_y_bench, &time);
    }

    err = clFinish(cqGpuCommandQueue);
    if(err != CL_SUCCESS) return err;

    double CSR5Spmv_time = 0;
    //anonymouslib_timer CSR5Spmv_timer;
    //CSR5Spmv_timer.start();

    // time spmv by running NUM_RUN times
    for (int i = 0; i < NUM_RUN; i++)
    {
        err = A.spmv(alpha, d_y_bench, &time);
        CSR5Spmv_time += time;
    }
    err = clFinish(cqGpuCommandQueue);
    //if(err != CL_SUCCESS) return err;

    //double CSR5Spmv_time = CSR5Spmv_timer.stop() / (double)NUM_RUN;
    CSR5Spmv_time = CSR5Spmv_time / (double)NUM_RUN;

    if (NUM_RUN)
        cout << "CSR5-based SpMV time = " << CSR5Spmv_time
             << " ms. Bandwidth = " << gb/(1.0e+6 * CSR5Spmv_time)
             << " GB/s. GFlops = " << gflop/(1.0e+6 * CSR5Spmv_time)  << " GFlops." << endl;

    A.destroy();

    if(d_csrRowPtrA) err = clReleaseMemObject(d_csrRowPtrA); if(err != CL_SUCCESS) return err;
    if(d_csrColIdxA) err = clReleaseMemObject(d_csrColIdxA); if(err != CL_SUCCESS) return err;
    if(d_csrValA) err = clReleaseMemObject(d_csrValA); if(err != CL_SUCCESS) return err;
    if(d_x) err = clReleaseMemObject(d_x); if(err != CL_SUCCESS) return err;
    if(d_y) err = clReleaseMemObject(d_y); if(err != CL_SUCCESS) return err;
    if(d_y_bench) err = clReleaseMemObject(d_y_bench); if(err != CL_SUCCESS) return err;

    return err;
}

int main(int argc, char ** argv)
{
    int m, n, nnzA;
    int *csrRowPtrA;
    int *csrColIdxA;
    VALUE_TYPE *csrValA;

    // report precision of floating-point
    cout << "------------------------------------------------------" << endl;
    char  *precision;
    if (sizeof(VALUE_TYPE) == 4)
    {
        precision = "32-bit Single Precision";
    }
    else if (sizeof(VALUE_TYPE) == 8)
    {
        precision = "64-bit Double Precision";
    }
    else
    {
        cout << "Wrong precision. Program exit!" << endl;
        return 0;
    }

    cout << "PRECISION = " << precision << endl;
    cout << "------------------------------------------------------" << endl;

    //ex: ./spmv webbase-1M.mtx
    int argi = 1;

    char  *filename;
    if(argc > argi)
    {
        filename = argv[argi];
        argi++;
    }
    cout << "--------------" << filename << "--------------" << endl;

    // read matrix from mtx file
    int ret_code;
    MM_typecode matcode;
    FILE *f;

    int nnzA_mtx_report;
    int isInteger = 0, isReal = 0, isPattern = 0, isSymmetric = 0;

    // load matrix
    if ((f = fopen(filename, "r")) == NULL)
        return -1;

    if (mm_read_banner(f, &matcode) != 0)
    {
        cout << "Could not process Matrix Market banner." << endl;
        return -2;
    }

    if ( mm_is_complex( matcode ) )
    {
        cout <<"Sorry, data type 'COMPLEX' is not supported. " << endl;
        return -3;
    }

    if ( mm_is_pattern( matcode ) )  { isPattern = 1; /*cout << "type = Pattern" << endl;*/ }
    if ( mm_is_real ( matcode) )     { isReal = 1; /*cout << "type = real" << endl;*/ }
    if ( mm_is_integer ( matcode ) ) { isInteger = 1; /*cout << "type = integer" << endl;*/ }

    /* find out size of sparse matrix .... */
    ret_code = mm_read_mtx_crd_size(f, &m, &n, &nnzA_mtx_report);
    if (ret_code != 0)
        return -4;

    if ( mm_is_symmetric( matcode ) || mm_is_hermitian( matcode ) )
    {
        isSymmetric = 1;
        //cout << "symmetric = true" << endl;
    }
    else
    {
        //cout << "symmetric = false" << endl;
    }

    int *csrRowPtrA_counter = (int *)malloc((m+1) * sizeof(int));
    memset(csrRowPtrA_counter, 0, (m+1) * sizeof(int));

    int *csrRowIdxA_tmp = (int *)malloc(nnzA_mtx_report * sizeof(int));
    int *csrColIdxA_tmp = (int *)malloc(nnzA_mtx_report * sizeof(int));
    VALUE_TYPE *csrValA_tmp    = (VALUE_TYPE *)malloc(nnzA_mtx_report * sizeof(VALUE_TYPE));

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (int i = 0; i < nnzA_mtx_report; i++)
    {
        int idxi, idxj;
        double fval;
        int ival;

        if (isReal)
            fscanf(f, "%d %d %lg\n", &idxi, &idxj, &fval);
        else if (isInteger)
        {
            fscanf(f, "%d %d %d\n", &idxi, &idxj, &ival);
            fval = ival;
        }
        else if (isPattern)
        {
            fscanf(f, "%d %d\n", &idxi, &idxj);
            fval = 1.0;
        }

        // adjust from 1-based to 0-based
        idxi--;
        idxj--;

        csrRowPtrA_counter[idxi]++;
        csrRowIdxA_tmp[i] = idxi;
        csrColIdxA_tmp[i] = idxj;
        csrValA_tmp[i] = fval;
    }

    if (f != stdin)
        fclose(f);

    if (isSymmetric)
    {
        for (int i = 0; i < nnzA_mtx_report; i++)
        {
            if (csrRowIdxA_tmp[i] != csrColIdxA_tmp[i])
                csrRowPtrA_counter[csrColIdxA_tmp[i]]++;
        }
    }

    // exclusive scan for csrRowPtrA_counter
    int old_val, new_val;

    old_val = csrRowPtrA_counter[0];
    csrRowPtrA_counter[0] = 0;
    for (int i = 1; i <= m; i++)
    {
        new_val = csrRowPtrA_counter[i];
        csrRowPtrA_counter[i] = old_val + csrRowPtrA_counter[i-1];
        old_val = new_val;
    }

    nnzA = csrRowPtrA_counter[m];
    csrRowPtrA = (int *)malloc((m+1) * sizeof(int));
    memcpy(csrRowPtrA, csrRowPtrA_counter, (m+1) * sizeof(int));
    memset(csrRowPtrA_counter, 0, (m+1) * sizeof(int));

    csrColIdxA = (int *)malloc(nnzA * sizeof(int));
    csrValA    = (VALUE_TYPE *)malloc(nnzA * sizeof(VALUE_TYPE));

    if (isSymmetric)
    {
        for (int i = 0; i < nnzA_mtx_report; i++)
        {
            if (csrRowIdxA_tmp[i] != csrColIdxA_tmp[i])
            {
                int offset = csrRowPtrA[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
                csrColIdxA[offset] = csrColIdxA_tmp[i];
                csrValA[offset] = csrValA_tmp[i];
                csrRowPtrA_counter[csrRowIdxA_tmp[i]]++;

                offset = csrRowPtrA[csrColIdxA_tmp[i]] + csrRowPtrA_counter[csrColIdxA_tmp[i]];
                csrColIdxA[offset] = csrRowIdxA_tmp[i];
                csrValA[offset] = csrValA_tmp[i];
                csrRowPtrA_counter[csrColIdxA_tmp[i]]++;
            }
            else
            {
                int offset = csrRowPtrA[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
                csrColIdxA[offset] = csrColIdxA_tmp[i];
                csrValA[offset] = csrValA_tmp[i];
                csrRowPtrA_counter[csrRowIdxA_tmp[i]]++;
            }
        }
    }
    else
    {
        for (int i = 0; i < nnzA_mtx_report; i++)
        {
            int offset = csrRowPtrA[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
            csrColIdxA[offset] = csrColIdxA_tmp[i];
            csrValA[offset] = csrValA_tmp[i];
            csrRowPtrA_counter[csrRowIdxA_tmp[i]]++;
        }
    }

    // free tmp space
    free(csrColIdxA_tmp);
    free(csrValA_tmp);
    free(csrRowIdxA_tmp);
    free(csrRowPtrA_counter);

    srand(time(NULL));

    // set csrValA to 1, easy for checking floating-point results
    for (int i = 0; i < nnzA; i++)
    {
        csrValA[i] = rand() % 10;
    }

    cout << " ( " << m << ", " << n << " ) nnz = " << nnzA << endl;

    VALUE_TYPE *x = (VALUE_TYPE *)malloc(n * sizeof(VALUE_TYPE));
    for (int i = 0; i < n; i++)
        x[i] = rand() % 10;

    VALUE_TYPE *y = (VALUE_TYPE *)malloc(m * sizeof(VALUE_TYPE));
    VALUE_TYPE *y_ref = (VALUE_TYPE *)malloc(m * sizeof(VALUE_TYPE));

    double gb = getB<int, VALUE_TYPE>(m, nnzA);
    double gflop = getFLOP<int>(nnzA);

    VALUE_TYPE alpha = 1.0;

    // compute reference results on a cpu core
    anonymouslib_timer ref_timer;
    ref_timer.start();

    int ref_iter = 1;
    for (int iter = 0; iter < ref_iter; iter++)
    {
        for (int i = 0; i < m; i++)
        {
            VALUE_TYPE sum = 0;
            for (int j = csrRowPtrA[i]; j < csrRowPtrA[i+1]; j++)
                sum += x[csrColIdxA[j]] * csrValA[j] * alpha;
            y_ref[i] = sum;
        }
    }

    double ref_time = ref_timer.stop() / (double)ref_iter;
    cout << "cpu sequential time = " << ref_time
         << " ms. Bandwidth = " << gb/(1.0e+6 * ref_time)
         << " GB/s. GFlops = " << gflop/(1.0e+6 * ref_time)  << " GFlops." << endl << endl;

    // launch compute
    call_anonymouslib(m, n, nnzA, csrRowPtrA, csrColIdxA, csrValA, x, y, alpha);

    // compare reference and anonymouslib results
    int error_count = 0;
    for (int i = 0; i < m; i++)
        if (fabs(y_ref[i] - y[i]) > 0.01 * fabs(y_ref[i]))
        {
            error_count++;
//            cout << "ROW [ " << i << " ], NNZ SPAN: "
//                 << csrRowPtrA[i] << " - "
//                 << csrRowPtrA[i+1]
//                 << "\t ref = " << y_ref[i]
//                 << ", \t csr5 = " << y[i]
//                 << ", \t error = " << y_ref[i] - y[i]
//                 << endl;
//            break;

//            //if (fabs(y_ref[i] - y[i]) > 0.00001)
//            //    cout << ", \t error = " << y_ref[i] - y[i] << endl;
//            //else
//            //    cout << ". \t CORRECT!" << endl;
        }

    if (error_count == 0)
        cout << "Check... PASS!" << endl;
    else
        cout << "Check... NO PASS! #Error = " << error_count << " out of " << m << " entries." << endl;

    cout << "------------------------------------------------------" << endl;

    free(csrRowPtrA);
    free(csrColIdxA);
    free(csrValA);
    free(x);
    free(y);
    free(y_ref);

    return 0;
}

