#ifndef ANONYMOUSLIB_CUDA_H
#define ANONYMOUSLIB_CUDA_H

#include "detail/utils.h"
#include "detail/cuda/utils_cuda.h"

#include "detail/cuda/common_cuda.h"
#include "detail/cuda/format_cuda.h"
#include "detail/cuda/csr5_spmv_cuda.h"

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
    cudaTextureObject_t  _x_tex;
};

template <class ANONYMOUSLIB_IT, class ANONYMOUSLIB_UIT, class ANONYMOUSLIB_VT>
int anonymouslibHandle<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>::warmup()
{
    return format_warmup();
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
        checkCudaErrors(cudaFree(_csr5_partition_pointer));
        checkCudaErrors(cudaFree(_csr5_partition_descriptor));
        checkCudaErrors(cudaFree(_temp_calibrator));
        checkCudaErrors(cudaFree(_csr5_partition_descriptor_offset_pointer));
        if (_num_offsets) checkCudaErrors(cudaFree(_csr5_partition_descriptor_offset));

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
        cout << "omega = " << ANONYMOUSLIB_CSR5_OMEGA << ", sigma = " << _csr5_sigma << ". " << endl;

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
        //cout << "#partition = " << _p << endl;

        malloc_timer.start();
        // malloc the newly added arrays for CSR5
        checkCudaErrors(cudaMalloc((void **)&_csr5_partition_pointer, (_p + 1) * sizeof(ANONYMOUSLIB_UIT)));

        checkCudaErrors(cudaMalloc((void **)&_csr5_partition_descriptor, _p * ANONYMOUSLIB_CSR5_OMEGA * _num_packet * sizeof(ANONYMOUSLIB_UIT)));
        checkCudaErrors(cudaMemset(_csr5_partition_descriptor, 0, _p * ANONYMOUSLIB_CSR5_OMEGA * _num_packet * sizeof(ANONYMOUSLIB_UIT)));

        checkCudaErrors(cudaMalloc((void **)&_temp_calibrator, _p * sizeof(ANONYMOUSLIB_VT)));
        checkCudaErrors(cudaMemset(_temp_calibrator, 0, _p * sizeof(ANONYMOUSLIB_VT)));

        checkCudaErrors(cudaMalloc((void **)&_csr5_partition_descriptor_offset_pointer, (_p + 1) * sizeof(ANONYMOUSLIB_IT)));
        checkCudaErrors(cudaMemset(_csr5_partition_descriptor_offset_pointer, 0, (_p + 1) * sizeof(ANONYMOUSLIB_IT)));
        malloc_time += malloc_timer.stop();

        // convert csr data to csr5 data (3 steps)
        // step 1. generate partition pointer
        tile_ptr_timer.start();
        err = generate_partition_pointer(_csr5_sigma, _p, _m, _nnz,
                                         _csr5_partition_pointer, _csr_row_pointer);
        if (err != ANONYMOUSLIB_SUCCESS)
            return ANONYMOUSLIB_CSR_TO_CSR5_FAILED;
        cudaDeviceSynchronize();
        tile_ptr_time += tile_ptr_timer.stop();

        malloc_timer.start();
        ANONYMOUSLIB_UIT tail;
        checkCudaErrors(cudaMemcpy(&tail, &_csr5_partition_pointer[_p-1], sizeof(ANONYMOUSLIB_UIT),   cudaMemcpyDeviceToHost));
        _tail_partition_start = (tail << 1) >> 1;
        //cout << "_tail_partition_start = " << _tail_partition_start << endl;
        malloc_time += malloc_timer.stop();

        // step 2. generate partition descriptor

        _num_offsets = 0;
        tile_desc_timer.start();
        err = generate_partition_descriptor(_csr5_sigma, _p, _m,
                                            _bit_y_offset, _bit_scansum_offset, _num_packet,
                                            _csr_row_pointer, _csr5_partition_pointer, _csr5_partition_descriptor,
                                            _csr5_partition_descriptor_offset_pointer, &_num_offsets);
        if (err != ANONYMOUSLIB_SUCCESS)
            return ANONYMOUSLIB_CSR_TO_CSR5_FAILED;
        cudaDeviceSynchronize();
        tile_desc_time += tile_desc_timer.stop(); // fixed a bug here (April 2016)

        if (_num_offsets)
        {
            //cout << "has empty rows, _num_offsets = " << _num_offsets << endl;
            malloc_timer.start();
            checkCudaErrors(cudaMalloc((void **)&_csr5_partition_descriptor_offset, _num_offsets * sizeof(ANONYMOUSLIB_IT)));
            malloc_time += malloc_timer.stop();

            tile_desc_timer.start();
            err = generate_partition_descriptor_offset(_csr5_sigma, _p,
                                                _bit_y_offset, _bit_scansum_offset, _num_packet,
                                                _csr_row_pointer, _csr5_partition_pointer, _csr5_partition_descriptor,
                                                _csr5_partition_descriptor_offset_pointer, _csr5_partition_descriptor_offset);
            if (err != ANONYMOUSLIB_SUCCESS)
                return ANONYMOUSLIB_CSR_TO_CSR5_FAILED;
            cudaDeviceSynchronize();
            tile_desc_time += tile_desc_timer.stop();
        }

        // step 3. transpose column_index and value arrays
        transpose_timer.start();
        err = aosoa_transpose(_csr5_sigma, _nnz,
                              _csr5_partition_pointer, _csr_column_index, _csr_value, true);
        if (err != ANONYMOUSLIB_SUCCESS)
            return ANONYMOUSLIB_CSR_TO_CSR5_FAILED;
        cudaDeviceSynchronize();
        transpose_time += transpose_timer.stop();

        cout << "CSR->CSR5 malloc time = " << malloc_time << " ms." << endl;
        cout << "CSR->CSR5 tile_ptr time = " << tile_ptr_time << " ms." << endl;
        cout << "CSR->CSR5 tile_desc time = " << tile_desc_time << " ms." << endl;
        cout << "CSR->CSR5 transpose time = " << transpose_time << " ms." << endl;

        _format = ANONYMOUSLIB_FORMAT_CSR5;
    }

    return err;
}

template <class ANONYMOUSLIB_IT, class ANONYMOUSLIB_UIT, class ANONYMOUSLIB_VT>
int anonymouslibHandle<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>::setX(ANONYMOUSLIB_VT *x)
{
    int err = ANONYMOUSLIB_SUCCESS;

    _x = x;

    // create texture object
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = _x;
    resDesc.res.linear.sizeInBytes = _n * sizeof(ANONYMOUSLIB_VT);
    if (sizeof(ANONYMOUSLIB_VT) == sizeof(float))
    {
        resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
        resDesc.res.linear.desc.x = 32; // bits per channel
    }
    else if (sizeof(ANONYMOUSLIB_VT) == sizeof(double))
    {
        resDesc.res.linear.desc.f = cudaChannelFormatKindSigned;
        resDesc.res.linear.desc.x = 32; // bits per channel
        resDesc.res.linear.desc.y = 32; // bits per channel
    }
    else
    {
        return ANONYMOUSLIB_UNSUPPORTED_VALUE_TYPE;
    }

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;

    // create texture object: we only have to do this once!
    _x_tex = 0;
    cudaCreateTextureObject(&_x_tex, &resDesc, &texDesc, NULL);

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
        csr5_spmv(_csr5_sigma, _p, _m,
                  _bit_y_offset, _bit_scansum_offset, _num_packet,
                  _csr_row_pointer, _csr_column_index, _csr_value,
                  _csr5_partition_pointer, _csr5_partition_descriptor,
                  _csr5_partition_descriptor_offset_pointer, _csr5_partition_descriptor_offset,
                  _temp_calibrator, _tail_partition_start,
                  alpha, _x, _x_tex, /*beta,*/ y);
    }

    return err;
}

template <class ANONYMOUSLIB_IT, class ANONYMOUSLIB_UIT, class ANONYMOUSLIB_VT>
int anonymouslibHandle<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>::destroy()
{
    cudaDestroyTextureObject(_x_tex);
    return asCSR();
}

template <class ANONYMOUSLIB_IT, class ANONYMOUSLIB_UIT, class ANONYMOUSLIB_VT>
void anonymouslibHandle<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>::setSigma(int sigma)
{
    if (sigma == ANONYMOUSLIB_AUTO_TUNED_SIGMA)
    {
        int r = 4;
        int s = 32;
        int t = 256;
        int u = 6;
        
        int nnz_per_row = _nnz / _m;
        if (nnz_per_row <= r)
            _csr5_sigma = r;
        else if (nnz_per_row > r && nnz_per_row <= s)
            _csr5_sigma = nnz_per_row;
        else if (nnz_per_row <= t && nnz_per_row > s)
            _csr5_sigma = s;
        else // nnz_per_row > t
            _csr5_sigma = u;
    }
    else
    {
        _csr5_sigma = sigma;
    }
}

template <class ANONYMOUSLIB_IT, class ANONYMOUSLIB_UIT, class ANONYMOUSLIB_VT>
int anonymouslibHandle<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>::computeSigma()
{
    return _csr5_sigma;
}

#endif // ANONYMOUSLIB_CUDA_H
