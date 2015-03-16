#ifndef ANONYMOUSLIB_AVX2_H
#define ANONYMOUSLIB_AVX2_H

#include "detail/utils.h"
#include "detail/avx2/utils_avx2.h"

#include "detail/avx2/common_avx2.h"
#include "detail/avx2/format_avx2.h"
#include "detail/avx2/csr5_spmv_avx2.h"

template <class ANONYMOUSLIB_IT, class ANONYMOUSLIB_UIT, class ANONYMOUSLIB_VT>
class anonymouslibHandle
{
public:
    anonymouslibHandle(ANONYMOUSLIB_IT m, ANONYMOUSLIB_IT n) { _m = m; _n = n; }
    int warmup();
    int inputCSR(ANONYMOUSLIB_IT  nnz, ANONYMOUSLIB_IT *csr_row_pointer, ANONYMOUSLIB_IT *csr_column_index, ANONYMOUSLIB_VT *csr_value);
    int asCSR();
    int asCSR5();
    int setX(ANONYMOUSLIB_VT *x);
    int spmv(const ANONYMOUSLIB_VT alpha, ANONYMOUSLIB_VT *y);
    int destroy();
    void setSigma(int sigma);

private:
    int computeSigma();
    int _format;
    ANONYMOUSLIB_IT _m;
    ANONYMOUSLIB_IT _n;
    ANONYMOUSLIB_IT _nnz;

    ANONYMOUSLIB_IT *_csr_row_pointer;
    ANONYMOUSLIB_IT *_csr_column_index;
    ANONYMOUSLIB_VT *_csr_value;

    int         _csr5_sigma;
    int         _bit_y_offset;
    int         _bit_scansum_offset;
    int         _num_packet;
    ANONYMOUSLIB_IT _tail_partition_start;

    ANONYMOUSLIB_IT _p;
    ANONYMOUSLIB_UIT *_csr5_partition_pointer;
    ANONYMOUSLIB_UIT *_csr5_partition_descriptor;

    ANONYMOUSLIB_IT   _num_offsets;
    ANONYMOUSLIB_IT  *_csr5_partition_descriptor_offset_pointer;
    ANONYMOUSLIB_IT  *_csr5_partition_descriptor_offset;
    ANONYMOUSLIB_VT  *_temp_calibrator;

    ANONYMOUSLIB_VT         *_x;
};

template <class ANONYMOUSLIB_IT, class ANONYMOUSLIB_UIT, class ANONYMOUSLIB_VT>
int anonymouslibHandle<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>::warmup()
{
    return ANONYMOUSLIB_SUCCESS;
}

template <class ANONYMOUSLIB_IT, class ANONYMOUSLIB_UIT, class ANONYMOUSLIB_VT>
int anonymouslibHandle<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>::inputCSR(ANONYMOUSLIB_IT  nnz,
                                                                     ANONYMOUSLIB_IT *csr_row_pointer,
                                                                     ANONYMOUSLIB_IT *csr_column_index,
                                                                     ANONYMOUSLIB_VT *csr_value)
{
    _format = ANONYMOUSLIB_FORMAT_CSR;

    _nnz = nnz;

    _csr_row_pointer  = csr_row_pointer;
    _csr_column_index = csr_column_index;
    _csr_value        = csr_value;

    return ANONYMOUSLIB_SUCCESS;
}

template <class ANONYMOUSLIB_IT, class ANONYMOUSLIB_UIT, class ANONYMOUSLIB_VT>
int anonymouslibHandle<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>::asCSR()
{
    int err = ANONYMOUSLIB_SUCCESS;

    if (_format == ANONYMOUSLIB_FORMAT_CSR)
        return err;

    if (_format == ANONYMOUSLIB_FORMAT_CSR5)
    {
        // convert csr5 data to csr data
        err = aosoa_transpose(_csr5_sigma, _nnz,
                              _csr5_partition_pointer, _csr_column_index, _csr_value, false);

        // free the two newly added CSR5 arrays
        _mm_free(_csr5_partition_pointer);
        _mm_free(_csr5_partition_descriptor);
        _mm_free(_temp_calibrator);
        _mm_free(_csr5_partition_descriptor_offset_pointer);
        if (_num_offsets) _mm_free(_csr5_partition_descriptor_offset);

        _format = ANONYMOUSLIB_FORMAT_CSR;
    }

    return err;
}

template <class ANONYMOUSLIB_IT, class ANONYMOUSLIB_UIT, class ANONYMOUSLIB_VT>
int anonymouslibHandle<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>::asCSR5()
{
    int err = ANONYMOUSLIB_SUCCESS;

    if (_format == ANONYMOUSLIB_FORMAT_CSR5)
        return err;

    if (_format == ANONYMOUSLIB_FORMAT_CSR)
    {
        double malloc_time = 0, tile_ptr_time = 0, tile_desc_time = 0, transpose_time = 0;
        anonymouslib_timer malloc_timer, tile_ptr_timer, tile_desc_timer, transpose_timer;

        // compute sigma
        _csr5_sigma = computeSigma();
        cout << "omega = " << ANONYMOUSLIB_CSR5_OMEGA << ", sigma = " << _csr5_sigma << ". ";

        // compute how many bits required for `y_offset' and `carry_offset'
        int base = 2;
        _bit_y_offset = 1;
        while (base < ANONYMOUSLIB_CSR5_OMEGA * _csr5_sigma) { base *= 2; _bit_y_offset++; }

        base = 2;
        _bit_scansum_offset = 1;
        while (base < ANONYMOUSLIB_CSR5_OMEGA) { base *= 2; _bit_scansum_offset++; }

        if (_bit_y_offset + _bit_scansum_offset > sizeof(ANONYMOUSLIB_UIT) * 8 - 1) //the 1st bit of bit-flag should be in the first packet
            return ANONYMOUSLIB_UNSUPPORTED_CSR5_OMEGA;

        int bit_all = _bit_y_offset + _bit_scansum_offset + _csr5_sigma;
        _num_packet = ceil((double)bit_all / (double)(sizeof(ANONYMOUSLIB_UIT) * 8));

        // calculate the number of partitions
        _p = ceil((double)_nnz / (double)(ANONYMOUSLIB_CSR5_OMEGA * _csr5_sigma));
        cout << "#partition = " << _p << endl;

        malloc_timer.start();
        // malloc the newly added arrays for CSR5
        _csr5_partition_pointer = (ANONYMOUSLIB_UIT *)_mm_malloc((_p + 1) * sizeof(ANONYMOUSLIB_UIT), ANONYMOUSLIB_X86_CACHELINE);

        _csr5_partition_descriptor = (ANONYMOUSLIB_UIT *)_mm_malloc(_p * ANONYMOUSLIB_CSR5_OMEGA * _num_packet * sizeof(ANONYMOUSLIB_UIT), ANONYMOUSLIB_X86_CACHELINE);
        memset(_csr5_partition_descriptor, 0, _p * ANONYMOUSLIB_CSR5_OMEGA * _num_packet * sizeof(ANONYMOUSLIB_UIT));

        int num_thread = omp_get_max_threads();
        _temp_calibrator = (ANONYMOUSLIB_VT *)_mm_malloc(num_thread * ANONYMOUSLIB_X86_CACHELINE, ANONYMOUSLIB_X86_CACHELINE);
        memset(_temp_calibrator, 0, num_thread * ANONYMOUSLIB_X86_CACHELINE);

        _csr5_partition_descriptor_offset_pointer = (ANONYMOUSLIB_IT *)_mm_malloc((_p + 1) * sizeof(ANONYMOUSLIB_IT), ANONYMOUSLIB_X86_CACHELINE);
        memset(_csr5_partition_descriptor_offset_pointer, 0, (_p + 1) * sizeof(ANONYMOUSLIB_IT));
        malloc_time += malloc_timer.stop();

        // convert csr data to csr5 data (3 steps)
        // step 1. generate partition pointer
        tile_ptr_timer.start();
        err = generate_partition_pointer<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT>(_csr5_sigma, _p, _m, _nnz,
                                         _csr5_partition_pointer, _csr_row_pointer);
        if (err != ANONYMOUSLIB_SUCCESS)
            return ANONYMOUSLIB_CSR_TO_CSR5_FAILED;
        tile_ptr_time += tile_ptr_timer.stop();

        _tail_partition_start = (_csr5_partition_pointer[_p-1] << 1) >> 1;
        //cout << "_tail_partition_start = " << _tail_partition_start << endl;

        // step 2. generate partition descriptor
        tile_desc_timer.start();
        _num_offsets = 0;
        err = generate_partition_descriptor<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT>(_csr5_sigma, _p, _m,
                                            _bit_y_offset, _bit_scansum_offset, _num_packet,
                                            _csr_row_pointer, _csr5_partition_pointer, _csr5_partition_descriptor,
                                            _csr5_partition_descriptor_offset_pointer, &_num_offsets);
        if (err != ANONYMOUSLIB_SUCCESS)
            return ANONYMOUSLIB_CSR_TO_CSR5_FAILED;
        tile_desc_time += tile_desc_timer.stop();

        if (_num_offsets)
        {
            //cout << "has empty rows, _num_offsets = " << _num_offsets << endl;
            malloc_timer.start();
            _csr5_partition_descriptor_offset = (ANONYMOUSLIB_IT *)_mm_malloc(_num_offsets * sizeof(ANONYMOUSLIB_IT), ANONYMOUSLIB_X86_CACHELINE);
            //memset(_csr5_partition_descriptor_offset, 0, _num_offsets * sizeof(ANONYMOUSLIB_IT));
            malloc_time += malloc_timer.stop();

            tile_desc_timer.start();
            err = generate_partition_descriptor_offset<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT>(_csr5_sigma, _p,
                                                _bit_y_offset, _bit_scansum_offset, _num_packet,
                                                _csr_row_pointer, _csr5_partition_pointer, _csr5_partition_descriptor,
                                                _csr5_partition_descriptor_offset_pointer, _csr5_partition_descriptor_offset);

            //for (int i = 0; i < _num_offsets; i++)
            //    cout << "_csr5_partition_descriptor_offset = " << _csr5_partition_descriptor_offset[i] << endl;
            if (err != ANONYMOUSLIB_SUCCESS)
                return ANONYMOUSLIB_CSR_TO_CSR5_FAILED;
            tile_desc_time += tile_desc_timer.stop();
        }

        // step 3. transpose column_index and value arrays
        transpose_timer.start();
        err = aosoa_transpose(_csr5_sigma, _nnz,
                              _csr5_partition_pointer, _csr_column_index, _csr_value, true);
        if (err != ANONYMOUSLIB_SUCCESS)
            return ANONYMOUSLIB_CSR_TO_CSR5_FAILED;
        transpose_time += transpose_timer.stop();

        printf("CSR->CSR5 malloc time = %f ms\n", malloc_time);
        printf("CSR->CSR5 tile_ptr time = %f ms\n", tile_ptr_time);
        printf("CSR->CSR5 tile_desc time = %f ms\n", tile_desc_time);
        printf("CSR->CSR5 transpose time = %f ms\n", transpose_time);

        _format = ANONYMOUSLIB_FORMAT_CSR5;
    }

    return err;
}

template <class ANONYMOUSLIB_IT, class ANONYMOUSLIB_UIT, class ANONYMOUSLIB_VT>
int anonymouslibHandle<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>::setX(ANONYMOUSLIB_VT *x)
{
    int err = ANONYMOUSLIB_SUCCESS;

    _x = x;

    return err;
}

template <class ANONYMOUSLIB_IT, class ANONYMOUSLIB_UIT, class ANONYMOUSLIB_VT>
int anonymouslibHandle<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>::spmv(const ANONYMOUSLIB_VT  alpha,
                                                                 ANONYMOUSLIB_VT       *y)
{
    int err = ANONYMOUSLIB_SUCCESS;

    if (_format == ANONYMOUSLIB_FORMAT_CSR)
    {
        return ANONYMOUSLIB_UNSUPPORTED_CSR_SPMV;
    }

    if (_format == ANONYMOUSLIB_FORMAT_CSR5)
    {
        csr5_spmv<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>(_csr5_sigma, _p, _m,
                  _bit_y_offset, _bit_scansum_offset, _num_packet,
                  _csr_row_pointer, _csr_column_index, _csr_value,
                  _csr5_partition_pointer, _csr5_partition_descriptor,
                  _csr5_partition_descriptor_offset_pointer, _csr5_partition_descriptor_offset,
                  _temp_calibrator, _tail_partition_start,
                  alpha, _x, y);
    }

    return err;
}

template <class ANONYMOUSLIB_IT, class ANONYMOUSLIB_UIT, class ANONYMOUSLIB_VT>
int anonymouslibHandle<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>::destroy()
{
    return asCSR();
}

template <class ANONYMOUSLIB_IT, class ANONYMOUSLIB_UIT, class ANONYMOUSLIB_VT>
void anonymouslibHandle<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>::setSigma(int sigma)
{
    _csr5_sigma = sigma;
}

template <class ANONYMOUSLIB_IT, class ANONYMOUSLIB_UIT, class ANONYMOUSLIB_VT>
int anonymouslibHandle<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>::computeSigma()
{
    return _csr5_sigma;
}

#endif // ANONYMOUSLIB_AVX2_H
