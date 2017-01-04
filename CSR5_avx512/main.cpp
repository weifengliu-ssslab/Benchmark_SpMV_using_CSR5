#include <stdlib.h>
#include <stdio.h>
#include <offload.h>

#include <iostream>
#include <cmath>

#include "anonymouslib_avx512.h"

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
                  VALUE_TYPE *x, VALUE_TYPE *y, VALUE_TYPE alpha, char *filename)
{
    int err = 0;

    memset(y, 0, sizeof(VALUE_TYPE) * m);

    VALUE_TYPE *y_bench = (VALUE_TYPE *)_mm_malloc(sizeof(VALUE_TYPE) * m, ANONYMOUSLIB_X86_CACHELINE);
    double gb = getB<int, VALUE_TYPE>(m, nnzA);
    double gflop = getFLOP<int>(nnzA);

    anonymouslib_timer asCSR5_timer;
    anonymouslib_timer CSR5Spmv_timer;
    anonymouslib_timer ref_timer;

    printf("omp_get_max_threads = %i\n", omp_get_max_threads());

    /*
    ref_timer.start();

    int ref_iter = 1000;
    #pragma omp parallel for
    for (int iter = 0; iter < ref_iter; iter++)
    {
        for (int i = 0; i < m; i++)
        {
            VALUE_TYPE sum = 0;
            for (int j = csrRowPtrA[i]; j < csrRowPtrA[i+1]; j++)
                sum += x[csrColIdxA[j]] * csrValA[j];
            y_bench[i] = sum;
        }
    }

    double ref_time = ref_timer.stop() / (double)ref_iter;

    printf("CSR-based SpMV OMP time = %f ms. Bandwidth = %f GB/s. GFlops = %f GFlops.\n\n",
               ref_time, gb/(1.0e+6 * ref_time), gflop/(1.0e+6 * ref_time));
    */
    anonymouslibHandle<int, unsigned int, VALUE_TYPE> A(m, n);
    err = A.inputCSR(nnzA, csrRowPtrA, csrColIdxA, csrValA);
    //cout << "inputCSR err = " << err << endl;

    err = A.setX(x); // you only need to do it once!
    //cout << "setX err = " << err << endl;

    int sigma = ANONYMOUSLIB_CSR5_SIGMA; //nnzA/(8*ANONYMOUSLIB_CSR5_OMEGA);
    A.setSigma(sigma);

    A.asCSR5();
    A.asCSR();

    // record a correct CSR->CSR5 time without PCIe overhead
    asCSR5_timer.start();
    err = A.asCSR5();
    printf("CSR->CSR5 time = %f ms.\n", asCSR5_timer.stop());
    //cout << "asCSR5 err = " << err << endl;
   
    // check correctness by running 1 time
    err = A.spmv(alpha, y);
    //cout << "spmv err = " << err << endl;

    double CSR5Spmv_gflops = 0;
    // warm up by running 50 times
    if (NUM_RUN)
    {
        for (int i = 0; i < 50; i++)
            err = A.spmv(alpha, y_bench);

        CSR5Spmv_timer.start();

        // time spmv by running NUM_RUN times
        for (int i = 0; i < NUM_RUN; i++)
            err = A.spmv(alpha, y_bench);

        double CSR5Spmv_time = CSR5Spmv_timer.stop() / (double)NUM_RUN;

        printf("CSR5-based SpMV AVX512 time = %f ms. Bandwidth = %f GB/s. GFlops = %f GFlops.\n",
               CSR5Spmv_time, gb/(1.0e+6 * CSR5Spmv_time), gflop/(1.0e+6 * CSR5Spmv_time));
        
        CSR5Spmv_gflops = gflop/(1.0e+6 * CSR5Spmv_time);
    }

    // write results to text (scv) file
    FILE *fout = fopen("results.csv", "a");
    if (fout == NULL)
        printf("Writing results fails.\n");
    fprintf(fout, "%s,%f\n", filename, CSR5Spmv_gflops);
    fclose(fout);

    _mm_free(y_bench);

    return err;
}

int main(int argc, char ** argv)
{
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

    int m, n, nnzA;
    int *csrRowPtrA;
    int *csrColIdxA;
    VALUE_TYPE *csrValA;

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
    int isInteger = 0, isReal = 0, isPattern = 0, isSymmetric = 0, isComplex = 0;

    // load matrix
    if ((f = fopen(filename, "r")) == NULL)
        return -1;

    if (mm_read_banner(f, &matcode) != 0)
    {
        cout << "Could not process Matrix Market banner." << endl;
        return -2;
    }

    if ( mm_is_pattern( matcode ) )  { isPattern = 1;} //cout << "type = Pattern" << endl;
    if ( mm_is_real ( matcode) )     { isReal = 1;} //cout << "type = real" << endl;
    if ( mm_is_complex( matcode ) ) { isComplex = 1; /*printf("type = real\n");*/ }
    if ( mm_is_integer ( matcode ) ) { isInteger = 1;} //cout << "type = integer" << endl

    // find out size of sparse matrix .... 
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

    // NOTE: when reading in doubles, ANSI C requires the use of the "l"  
    //   specifier as in "%lg", "%lf", "%le", otherwise errors will occur 
    //  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            

    for (int i = 0; i < nnzA_mtx_report; i++)
    {
        int idxi, idxj;
        double fval, fval_im;
        int ival;

        if (isReal)
            fscanf(f, "%d %d %lg\n", &idxi, &idxj, &fval);
        else if (isComplex)
        {
            fscanf(f, "%d %d %lg %lg\n", &idxi, &idxj, &fval, &fval_im);
        }
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
    csrRowPtrA = (int *)_mm_malloc((m+1) * sizeof(int), ANONYMOUSLIB_X86_CACHELINE);
    memcpy(csrRowPtrA, csrRowPtrA_counter, (m+1) * sizeof(int));
    memset(csrRowPtrA_counter, 0, (m+1) * sizeof(int));

    csrColIdxA = (int *)_mm_malloc(nnzA * sizeof(int), ANONYMOUSLIB_X86_CACHELINE);
    csrValA    = (VALUE_TYPE *)_mm_malloc(nnzA * sizeof(VALUE_TYPE), ANONYMOUSLIB_X86_CACHELINE);

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

    VALUE_TYPE *x = (VALUE_TYPE *)_mm_malloc(n * sizeof(VALUE_TYPE), ANONYMOUSLIB_X86_CACHELINE);
    for (int i = 0; i < n; i++)
        x[i] = rand() % 10;

    VALUE_TYPE *y = (VALUE_TYPE *)_mm_malloc(m * sizeof(VALUE_TYPE), ANONYMOUSLIB_X86_CACHELINE);
    VALUE_TYPE *y_ref = (VALUE_TYPE *)_mm_malloc(m * sizeof(VALUE_TYPE), ANONYMOUSLIB_X86_CACHELINE);

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
    cout << "cpu host-sequential time = " << ref_time
         << " ms. Bandwidth = " << gb/(1.0e+6 * ref_time)
         << " GB/s. GFlops = " << gflop/(1.0e+6 * ref_time)  << " GFlops." << endl << endl;

    // launch compute
    call_anonymouslib(m, n, nnzA, csrRowPtrA, csrColIdxA, csrValA, x, y, alpha, filename);

    // compare reference and anonymouslib results
    int error_count = 0;
    for (int i = 0; i < m; i++)
        if (abs(y_ref[i] - y[i]) > 0.01 * abs(y_ref[i]))
        {
            error_count++;
/*            cout << "ROW [ " << i << " ], NNZ SPAN: "
                 << csrRowPtrA[i] << " - "
                 << csrRowPtrA[i+1]
                 << "\t ref = " << y_ref[i]
                 << ", \t csr5 = " << y[i]
                 << ", \t error = " << y_ref[i] - y[i]
                 << endl;
*/
//            break;
        }

    if (error_count == 0)
        cout << "Check... PASS!" << endl;
    else
        cout << "Check... NO PASS! #Error = " << error_count << " out of " << m << " entries." << endl;

    cout << "------------------------------------------------------" << endl;

    _mm_free(csrRowPtrA);
    _mm_free(csrColIdxA);
    _mm_free(csrValA);
    _mm_free(x);
    _mm_free(y);
    _mm_free(y_ref);

    return 0;
}

