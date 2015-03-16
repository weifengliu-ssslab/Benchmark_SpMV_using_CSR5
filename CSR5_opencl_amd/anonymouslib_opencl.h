#ifndef ANONYMOUSLIB_OPENCL_H
#define ANONYMOUSLIB_OPENCL_H

#include "detail/utils.h"
#include "detail/opencl/utils_opencl.h"

#include "detail/opencl/common_opencl.h"
#include "detail/opencl/format_opencl.h"
#include "detail/opencl/csr5_spmv_opencl.h"

template <class ANONYMOUSLIB_IT, class ANONYMOUSLIB_UIT, class ANONYMOUSLIB_VT>
class anonymouslibHandle
{
public:
    anonymouslibHandle(ANONYMOUSLIB_IT m, ANONYMOUSLIB_IT n);
    int setOCLENV(cl_context _ocl_context, cl_command_queue _ocl_command_queue);
    int warmup();
    int inputCSR(ANONYMOUSLIB_IT  nnz, cl_mem csr_row_pointer, cl_mem csr_column_index, cl_mem csr_value);
    int asCSR();
    int asCSR5();
    int setX(cl_mem x);
    int spmv(const ANONYMOUSLIB_VT alpha, cl_mem y, double *time);
    int destroy();
    int setSigma(int sigma);

private:
    cl_context          _ocl_context;
    cl_command_queue    _ocl_command_queue;

    cl_program          _ocl_program_format;
    cl_kernel           _ocl_kernel_warmup;
    cl_kernel           _ocl_kernel_generate_partition_pointer_s1;
    cl_kernel           _ocl_kernel_generate_partition_pointer_s2;
    cl_kernel           _ocl_kernel_generate_partition_descriptor_s0;
    cl_kernel           _ocl_kernel_generate_partition_descriptor_s1;
    cl_kernel           _ocl_kernel_generate_partition_descriptor_s2;
    cl_kernel           _ocl_kernel_generate_partition_descriptor_s3;
    cl_kernel           _ocl_kernel_generate_partition_descriptor_offset;
    cl_kernel           _ocl_kernel_aosoa_transpose_smem_iT;
    cl_kernel           _ocl_kernel_aosoa_transpose_smem_vT;

    cl_program          _ocl_program_csr5_spmv;
    cl_kernel           _ocl_kernel_spmv_csr5_compute;
    cl_kernel           _ocl_kernel_spmv_csr5_calibrate;
    cl_kernel           _ocl_kernel_spmv_csr5_tail_partition;

    string _ocl_source_code_string_format_const;
    string _ocl_source_code_string_csr5_spmv_const;

    int computeSigma();
    int _format;
    ANONYMOUSLIB_IT _m;
    ANONYMOUSLIB_IT _n;
    ANONYMOUSLIB_IT _nnz;

    cl_mem _csr_row_pointer;
    cl_mem _csr_column_index;
    cl_mem _csr_value;

    int         _csr5_sigma;
    int         _bit_y_offset;
    int         _bit_scansum_offset;
    int         _num_packet;
    ANONYMOUSLIB_IT _tail_partition_start;

    ANONYMOUSLIB_IT _p;
    cl_mem _csr5_partition_pointer;
    cl_mem _csr5_partition_descriptor;

    ANONYMOUSLIB_IT   _num_offsets;
    cl_mem  _csr5_partition_descriptor_offset_pointer;
    cl_mem  _csr5_partition_descriptor_offset;
    cl_mem  _temp_calibrator;

    cl_mem         _x;
};

template <class ANONYMOUSLIB_IT, class ANONYMOUSLIB_UIT, class ANONYMOUSLIB_VT>
anonymouslibHandle<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>::anonymouslibHandle(ANONYMOUSLIB_IT         m,
                                                                       ANONYMOUSLIB_IT         n)
{
    _m = m;
    _n = n;
}

template <class ANONYMOUSLIB_IT, class ANONYMOUSLIB_UIT, class ANONYMOUSLIB_VT>
int anonymouslibHandle<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>::warmup()
{
    format_warmup(_ocl_kernel_warmup, _ocl_context, _ocl_command_queue);

    return ANONYMOUSLIB_SUCCESS;
}

template <class ANONYMOUSLIB_IT, class ANONYMOUSLIB_UIT, class ANONYMOUSLIB_VT>
int anonymouslibHandle<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>::inputCSR(ANONYMOUSLIB_IT  nnz,
                                                                     cl_mem csr_row_pointer,
                                                                     cl_mem csr_column_index,
                                                                     cl_mem csr_value)
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
        double time = 0;
        err = aosoa_transpose<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>
                (_ocl_kernel_aosoa_transpose_smem_iT, _ocl_kernel_aosoa_transpose_smem_vT, _ocl_command_queue,
                 _csr5_sigma, _nnz, _csr5_partition_pointer, _csr_column_index, _csr_value, 0, &time);

        // free the two newly added CSR5 arrays
        if(_csr5_partition_pointer) err = clReleaseMemObject(_csr5_partition_pointer); if(err != CL_SUCCESS) return err;
        if(_csr5_partition_descriptor) err = clReleaseMemObject(_csr5_partition_descriptor); if(err != CL_SUCCESS) return err;
        if(_temp_calibrator) err = clReleaseMemObject(_temp_calibrator); if(err != CL_SUCCESS) return err;
        if(_csr5_partition_descriptor_offset_pointer) err = clReleaseMemObject(_csr5_partition_descriptor_offset_pointer); if(err != CL_SUCCESS) return err;
        if(_csr5_partition_descriptor_offset) err = clReleaseMemObject(_csr5_partition_descriptor_offset); if(err != CL_SUCCESS) return err;

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
        double time = 0;

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
        //cout << "#num_packet = " << _num_packet << endl;

        // calculate the number of partitions
        _p = ceil((double)_nnz / (double)(ANONYMOUSLIB_CSR5_OMEGA * _csr5_sigma));
        //cout << "#partition = " << _p << endl;

        malloc_timer.start();
        // malloc the newly added arrays for CSR5
        _csr5_partition_pointer = clCreateBuffer(_ocl_context, CL_MEM_READ_WRITE, (_p + 1) * sizeof(ANONYMOUSLIB_UIT), NULL, &err);
        if(err != CL_SUCCESS) return err;

        _csr5_partition_descriptor = clCreateBuffer(_ocl_context, CL_MEM_READ_WRITE, _p * ANONYMOUSLIB_CSR5_OMEGA * _num_packet * sizeof(ANONYMOUSLIB_UIT), NULL, &err);
        if(err != CL_SUCCESS) return err;

        _temp_calibrator = clCreateBuffer(_ocl_context, CL_MEM_READ_WRITE, _p * sizeof(ANONYMOUSLIB_VT), NULL, &err);
        if(err != CL_SUCCESS) return err;

        _csr5_partition_descriptor_offset_pointer = clCreateBuffer(_ocl_context, CL_MEM_READ_WRITE, (_p + 1) * sizeof(ANONYMOUSLIB_IT), NULL, &err);
        if(err != CL_SUCCESS) return err;
        err = clFinish(_ocl_command_queue);
        malloc_time += malloc_timer.stop();

        // convert csr data to csr5 data (3 steps)
        // step 1. generate partition pointer
        //tile_ptr_timer.start();
        err = generate_partition_pointer<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT>
                (_ocl_kernel_generate_partition_pointer_s1, _ocl_kernel_generate_partition_pointer_s2, _ocl_command_queue,
                 _csr5_sigma, _p, _m, _nnz, _csr5_partition_pointer, _csr_row_pointer, &time);
        if (err != ANONYMOUSLIB_SUCCESS)
            return ANONYMOUSLIB_CSR_TO_CSR5_FAILED;
        //err = clFinish(_ocl_command_queue);
        //tile_ptr_time += tile_ptr_timer.stop();
        tile_ptr_time += time;

        malloc_timer.start();
        ANONYMOUSLIB_UIT tail;

        err = clEnqueueReadBuffer(_ocl_command_queue, _csr5_partition_pointer, CL_TRUE,
                                  (_p-1) * sizeof(ANONYMOUSLIB_UIT), sizeof(ANONYMOUSLIB_UIT), &tail, 0, NULL, NULL);
        if(err != CL_SUCCESS) return err;
        err = clFinish(_ocl_command_queue);
        malloc_time += malloc_timer.stop();

        _tail_partition_start = (tail << 1) >> 1;
        //cout << "_tail_partition_start = " << _tail_partition_start << endl;

        // step 2. generate partition descriptor
        //tile_desc_timer.start();
        _num_offsets = 0;
        err = generate_partition_descriptor<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT>
                (_ocl_kernel_generate_partition_descriptor_s0,
                 _ocl_kernel_generate_partition_descriptor_s1,
                 _ocl_kernel_generate_partition_descriptor_s2,
                 _ocl_kernel_generate_partition_descriptor_s3,
                 _ocl_command_queue,
                 _csr5_sigma, _p, _m,
                 _bit_y_offset, _bit_scansum_offset, _num_packet,
                 _csr_row_pointer, _csr5_partition_pointer, _csr5_partition_descriptor,
                 _csr5_partition_descriptor_offset_pointer, &_num_offsets, &time);
        if (err != ANONYMOUSLIB_SUCCESS)
            return ANONYMOUSLIB_CSR_TO_CSR5_FAILED;
        //err = clFinish(_ocl_command_queue);
        //tile_desc_time += tile_desc_timer.stop();
        tile_desc_time += time;

        if (_num_offsets)
        {
            //cout << "has empty rows, _num_offsets = " << _num_offsets << endl;

            malloc_timer.start();

            _csr5_partition_descriptor_offset = clCreateBuffer(_ocl_context, CL_MEM_READ_WRITE, _num_offsets * sizeof(ANONYMOUSLIB_IT), NULL, &err);
            if(err != CL_SUCCESS) return err;
            err = clFinish(_ocl_command_queue);
            malloc_time += malloc_timer.stop();

            //tile_desc_timer.start();
            err = generate_partition_descriptor_offset<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT>
                    (_ocl_kernel_generate_partition_descriptor_offset, _ocl_command_queue,
                     _csr5_sigma, _p,
                     _bit_y_offset, _bit_scansum_offset, _num_packet,
                     _csr_row_pointer, _csr5_partition_pointer, _csr5_partition_descriptor,
                     _csr5_partition_descriptor_offset_pointer, _csr5_partition_descriptor_offset, &time);
            if (err != ANONYMOUSLIB_SUCCESS)
                return ANONYMOUSLIB_CSR_TO_CSR5_FAILED;
            //err = clFinish(_ocl_command_queue);
            //tile_desc_time += tile_desc_timer.stop();
            tile_desc_time += time;
        }
        else
        {
            _csr5_partition_descriptor_offset = clCreateBuffer(_ocl_context, CL_MEM_READ_WRITE, 1 * sizeof(ANONYMOUSLIB_IT), NULL, &err);
            if(err != CL_SUCCESS) return err;
        }

        // step 3. transpose column_index and value arrays
        //transpose_timer.start();
        err = aosoa_transpose<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>
                (_ocl_kernel_aosoa_transpose_smem_iT, _ocl_kernel_aosoa_transpose_smem_vT, _ocl_command_queue,
                 _csr5_sigma, _nnz, _csr5_partition_pointer, _csr_column_index, _csr_value, 1, &time);
        if (err != ANONYMOUSLIB_SUCCESS)
            return ANONYMOUSLIB_CSR_TO_CSR5_FAILED;
        //err = clFinish(_ocl_command_queue);
        //transpose_time += transpose_timer.stop();
        transpose_time += time;

        cout << endl << "CSR->CSR5 malloc time = " << malloc_time << " ms." << endl;
        cout << "CSR->CSR5 tile_ptr time = " << tile_ptr_time << " ms." << endl;
        cout << "CSR->CSR5 tile_desc time = " << tile_desc_time << " ms." << endl;
        cout << "CSR->CSR5 transpose time = " << transpose_time << " ms." << endl;

        _format = ANONYMOUSLIB_FORMAT_CSR5;
    }

    return err;
}

template <class ANONYMOUSLIB_IT, class ANONYMOUSLIB_UIT, class ANONYMOUSLIB_VT>
int anonymouslibHandle<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>::setX(cl_mem x)
{
    int err = ANONYMOUSLIB_SUCCESS;

    _x = x;

    return err;
}

template <class ANONYMOUSLIB_IT, class ANONYMOUSLIB_UIT, class ANONYMOUSLIB_VT>
int anonymouslibHandle<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>::spmv(const ANONYMOUSLIB_VT  alpha,
                                                                 cl_mem y,
                                                                 double *time)
{
    int err = ANONYMOUSLIB_SUCCESS;

    if (_format == ANONYMOUSLIB_FORMAT_CSR)
    {
        return ANONYMOUSLIB_UNSUPPORTED_CSR_SPMV;
    }

    if (_format == ANONYMOUSLIB_FORMAT_CSR5)
    {
        csr5_spmv<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>
                (_ocl_kernel_spmv_csr5_compute,
                 _ocl_kernel_spmv_csr5_calibrate,
                 _ocl_kernel_spmv_csr5_tail_partition,
                 _ocl_command_queue,
                 _csr5_sigma, _p, _m,
                 _bit_y_offset, _bit_scansum_offset, _num_packet,
                 _csr_row_pointer, _csr_column_index, _csr_value,
                 _csr5_partition_pointer, _csr5_partition_descriptor,
                 _csr5_partition_descriptor_offset_pointer, _csr5_partition_descriptor_offset,
                 _temp_calibrator, _tail_partition_start,
                 alpha, _x, /*beta,*/ y, time);
    }

    return err;
}

template <class ANONYMOUSLIB_IT, class ANONYMOUSLIB_UIT, class ANONYMOUSLIB_VT>
int anonymouslibHandle<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>::destroy()
{
    return asCSR();
}

template <class ANONYMOUSLIB_IT, class ANONYMOUSLIB_UIT, class ANONYMOUSLIB_VT>
int anonymouslibHandle<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>::setSigma(int sigma)
{
    int err = ANONYMOUSLIB_SUCCESS;

    if (sigma == ANONYMOUSLIB_AUTO_TUNED_SIGMA)
    {
        int r = 4;
        int s = 7;
        int t = 256;
        int u = 4;
        
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

    char omega_str [3]; //supports up to omega = 999
    sprintf (omega_str, "%d", ANONYMOUSLIB_CSR5_OMEGA);
    char sigma_str [3]; //supports up to omega = 999
    sprintf (sigma_str, "%d", _csr5_sigma);
    char threadgroup_str [4]; //supports up to omega = 9999
    sprintf (threadgroup_str, "%d", ANONYMOUSLIB_THREAD_GROUP);
    char threadbunch_str [3]; //supports up to omega = 999
    sprintf (threadbunch_str, "%d", ANONYMOUSLIB_THREAD_BUNCH);
    //cout << "sigma_str = " << sigma_str << endl;

    char *it_str = " int ";
    char *uit_str = " unsigned int ";

    char *vt_str;
    if (sizeof(ANONYMOUSLIB_VT) == 8)
        vt_str = " double ";
    else if (sizeof(ANONYMOUSLIB_VT) == 4)
        vt_str = " float ";

    string ocl_source_code_string_format = _ocl_source_code_string_format_const;
    string ocl_source_code_string_csr5_spmv = _ocl_source_code_string_csr5_spmv_const;

    // replace 'omega_replace_str' to 'omega_str'
    const string omega_replace_str ("_REPLACE_ANONYMOUSLIB_CSR5_OMEGA_SEGMENT_");
    size_t omega_found = ocl_source_code_string_format.find(omega_replace_str);
    ocl_source_code_string_format.replace( omega_found, omega_replace_str.length(), omega_str);
    omega_found = ocl_source_code_string_csr5_spmv.find(omega_replace_str);
    ocl_source_code_string_csr5_spmv.replace( omega_found, omega_replace_str.length(), omega_str);

    // replace 'sigma_replace_str' to 'sigma_str'
    const string sigma_replace_str ("_REPLACE_ANONYMOUSLIB_CSR5_SIGMA_SEGMENT_");
    size_t sigma_found = ocl_source_code_string_format.find(sigma_replace_str);
    ocl_source_code_string_format.replace( sigma_found, sigma_replace_str.length(), sigma_str);
    sigma_found = ocl_source_code_string_csr5_spmv.find(sigma_replace_str);
    ocl_source_code_string_csr5_spmv.replace( sigma_found, sigma_replace_str.length(), sigma_str);

    // replace 'threadgroup_replace_str' to 'threadgroup_str'
    const string threadgroup_replace_str ("_REPLACE_ANONYMOUSLIB_THREAD_GROUP_SEGMENT_");
    size_t threadgroup_found = ocl_source_code_string_format.find(threadgroup_replace_str);
    ocl_source_code_string_format.replace( threadgroup_found, threadgroup_replace_str.length(), threadgroup_str);
    threadgroup_found = ocl_source_code_string_csr5_spmv.find(threadgroup_replace_str);
    ocl_source_code_string_csr5_spmv.replace( threadgroup_found, threadgroup_replace_str.length(), threadgroup_str);

    // replace 'threadbunch_replace_str' to 'threadbunch_str'
    const string threadbunch_replace_str ("_REPLACE_ANONYMOUSLIB_THREAD_BUNCH_SEGMENT_");
    size_t threadbunch_found = ocl_source_code_string_csr5_spmv.find(threadbunch_replace_str);
    ocl_source_code_string_csr5_spmv.replace( threadbunch_found, threadbunch_replace_str.length(), threadbunch_str);

    // replace 'it_replace_str' to 'it_str'
    const string it_replace_str ("_REPLACE_ANONYMOUSLIB_CSR5_INDEX_TYPE_SEGMENT_");
    size_t it_found = ocl_source_code_string_format.find(it_replace_str);
    ocl_source_code_string_format.replace( it_found, it_replace_str.length(), it_str);
    it_found = ocl_source_code_string_csr5_spmv.find(it_replace_str);
    ocl_source_code_string_csr5_spmv.replace( it_found, it_replace_str.length(), it_str);

    // replace 'uit_replace_str' to 'uit_str'
    const string uit_replace_str ("_REPLACE_ANONYMOUSLIB_CSR5_UNSIGNED_INDEX_TYPE_SEGMENT_");
    size_t uit_found = ocl_source_code_string_format.find(uit_replace_str);
    ocl_source_code_string_format.replace( uit_found, uit_replace_str.length(), uit_str);
    uit_found = ocl_source_code_string_csr5_spmv.find(uit_replace_str);
    ocl_source_code_string_csr5_spmv.replace( uit_found, uit_replace_str.length(), uit_str);

    // replace 'vt_replace_str' to 'vt_str'
    const string vt_replace_str ("_REPLACE_ANONYMOUSLIB_CSR5_VALUE_TYPE_SEGMENT_");
    size_t vt_found = ocl_source_code_string_format.find(vt_replace_str);
    ocl_source_code_string_format.replace( vt_found, vt_replace_str.length(), vt_str);
    vt_found = ocl_source_code_string_csr5_spmv.find(vt_replace_str);
    ocl_source_code_string_csr5_spmv.replace( vt_found, vt_replace_str.length(), vt_str);

    const char *ocl_source_code_format = ocl_source_code_string_format.c_str();
    const char *ocl_source_code_csr5_spmv = ocl_source_code_string_csr5_spmv.c_str();

    //cout << ocl_source_code_csr5_spmv << endl;

    // Create the program
    size_t source_size_format[] = { strlen(ocl_source_code_format)};
    _ocl_program_format = clCreateProgramWithSource(_ocl_context, 1, &ocl_source_code_format, source_size_format, &err);
    if(err != CL_SUCCESS) return err;
    size_t source_size_csr5_spmv[] = { strlen(ocl_source_code_csr5_spmv)};
    _ocl_program_csr5_spmv = clCreateProgramWithSource(_ocl_context, 1, &ocl_source_code_csr5_spmv, source_size_csr5_spmv, &err);
    if(err != CL_SUCCESS) return err;

    // Build the program
    err = clBuildProgram(_ocl_program_format, 0, NULL, NULL, NULL, NULL);
    if(err != CL_SUCCESS) return err;

    err = clBuildProgram(_ocl_program_csr5_spmv, 0, NULL, NULL, NULL, NULL);
    if(err != CL_SUCCESS) return err;

    // Create kernels
    _ocl_kernel_warmup = clCreateKernel(_ocl_program_format, "warmup_kernel", &err);
    if(err != CL_SUCCESS) return err;
    _ocl_kernel_generate_partition_pointer_s1 = clCreateKernel(_ocl_program_format, "generate_partition_pointer_s1_kernel", &err);
    if(err != CL_SUCCESS) return err;
    _ocl_kernel_generate_partition_pointer_s2 = clCreateKernel(_ocl_program_format, "generate_partition_pointer_s2_kernel", &err);
    if(err != CL_SUCCESS) return err;
    _ocl_kernel_generate_partition_descriptor_s0 = clCreateKernel(_ocl_program_format, "generate_partition_descriptor_s0_kernel", &err);
    if(err != CL_SUCCESS) return err;
    _ocl_kernel_generate_partition_descriptor_s1 = clCreateKernel(_ocl_program_format, "generate_partition_descriptor_s1_kernel", &err);
    if(err != CL_SUCCESS) return err;
    _ocl_kernel_generate_partition_descriptor_s2 = clCreateKernel(_ocl_program_format, "generate_partition_descriptor_s2_kernel", &err);
    if(err != CL_SUCCESS) return err;
    _ocl_kernel_generate_partition_descriptor_s3 = clCreateKernel(_ocl_program_format, "generate_partition_descriptor_s3_kernel", &err);
    if(err != CL_SUCCESS) return err;
    _ocl_kernel_generate_partition_descriptor_offset = clCreateKernel(_ocl_program_format, "generate_partition_descriptor_offset_kernel", &err);
    if(err != CL_SUCCESS) return err;
    _ocl_kernel_aosoa_transpose_smem_iT = clCreateKernel(_ocl_program_format, "aosoa_transpose_kernel_smem_iT", &err);
    if(err != CL_SUCCESS) return err;
    _ocl_kernel_aosoa_transpose_smem_vT = clCreateKernel(_ocl_program_format, "aosoa_transpose_kernel_smem_vT", &err);
    if(err != CL_SUCCESS) return err;

    _ocl_kernel_spmv_csr5_compute = clCreateKernel(_ocl_program_csr5_spmv, "spmv_csr5_compute_kernel", &err);
    if(err != CL_SUCCESS) return err;
    _ocl_kernel_spmv_csr5_calibrate = clCreateKernel(_ocl_program_csr5_spmv, "spmv_csr5_calibrate_kernel", &err);
    if(err != CL_SUCCESS) return err;
    _ocl_kernel_spmv_csr5_tail_partition = clCreateKernel(_ocl_program_csr5_spmv, "spmv_csr5_tail_partition_kernel", &err);
    if(err != CL_SUCCESS) return err;

    return err;
}

template <class ANONYMOUSLIB_IT, class ANONYMOUSLIB_UIT, class ANONYMOUSLIB_VT>
int anonymouslibHandle<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>::computeSigma()
{
    return _csr5_sigma;
}

template <class ANONYMOUSLIB_IT, class ANONYMOUSLIB_UIT, class ANONYMOUSLIB_VT>
int anonymouslibHandle<ANONYMOUSLIB_IT, ANONYMOUSLIB_UIT, ANONYMOUSLIB_VT>::setOCLENV(cl_context       ocl_context,
                                                                      cl_command_queue ocl_command_queue)
{
    int err = ANONYMOUSLIB_SUCCESS;

    _ocl_context = ocl_context;
    _ocl_command_queue = ocl_command_queue;

    _ocl_source_code_string_format_const =
    "    #pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable                                                                                                                       \n"
    "    #pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable                                                                                                                          \n"
    "                                                                                                                                                                                                \n"
    "    #pragma OPENCL EXTENSION cl_khr_fp64 : enable                                                                                                                                               \n"
    "                                                                                                                                                                                                \n"
    "    #define  ANONYMOUSLIB_CSR5_OMEGA _REPLACE_ANONYMOUSLIB_CSR5_OMEGA_SEGMENT_ \n"
    "    #define  ANONYMOUSLIB_CSR5_SIGMA _REPLACE_ANONYMOUSLIB_CSR5_SIGMA_SEGMENT_ \n"
    "    #define  ANONYMOUSLIB_THREAD_GROUP _REPLACE_ANONYMOUSLIB_THREAD_GROUP_SEGMENT_ \n"
    "    typedef _REPLACE_ANONYMOUSLIB_CSR5_INDEX_TYPE_SEGMENT_   iT; \n"
    "    typedef _REPLACE_ANONYMOUSLIB_CSR5_UNSIGNED_INDEX_TYPE_SEGMENT_   uiT; \n"
    "    typedef _REPLACE_ANONYMOUSLIB_CSR5_VALUE_TYPE_SEGMENT_   vT; \n"
    "                                                                                                                                                                                                \n"
    "    inline                                                                                                                                                                                      \n"
    "    iT binary_search_right_boundary_kernel(__global const iT *d_row_pointer,                                                                                                                    \n"
    "                                           const iT  key_input,                                                                                                                                 \n"
    "                                           const iT  size)                                                                                                                                      \n"
    "    {                                                                                                                                                                                           \n"
    "        iT start = 0;                                                                                                                                                                           \n"
    "        iT stop  = size - 1;                                                                                                                                                                    \n"
    "        iT median;                                                                                                                                                                              \n"
    "        iT key_median;                                                                                                                                                                          \n"
    "                                                                                                                                                                                                \n"
    "        while (stop >= start)                                                                                                                                                                   \n"
    "        {                                                                                                                                                                                       \n"
    "            median = (stop + start) / 2;                                                                                                                                                        \n"
    "            key_median = d_row_pointer[median];                                                                                                                                                 \n"
    "                                                                                                                                                                                                \n"
    "            if (key_input >= key_median)                                                                                                                                                        \n"
    "                start = median + 1;                                                                                                                                                             \n"
    "            else                                                                                                                                                                                \n"
    "                stop = median - 1;                                                                                                                                                              \n"
    "        }                                                                                                                                                                                       \n"
    "                                                                                                                                                                                                \n"
    "        return start;                                                                                                                                                                           \n"
    "    }                                                                                                                                                                                           \n"
    "                                                                                                                                                                                                \n"
    "                                                                                                                                                                                                \n"
    "    inline                                                                                                                                                                                      \n"
    "    void scan_64_plus1(__local volatile int *s_scan,                                                                                                                                            \n"
    "                       const int      lane_id)                                                                                                                                                  \n"
    "    {                                                                                                                                                                                           \n"
    "        int ai, bi;                                                                                                                                                                             \n"
    "        int baseai = 1 + 2 * lane_id;                                                                                                                                                           \n"
    "        int basebi = baseai + 1;                                                                                                                                                                \n"
    "        int temp;                                                                                                                                                                               \n"
    "                                                                                                                                                                                                \n"
    "        if (lane_id < 32) { ai = baseai - 1;     bi = basebi - 1;     s_scan[bi] += s_scan[ai]; }                                                                                               \n"
    "        if (lane_id < 16) { ai =  2 * baseai - 1;  bi =  2 * basebi - 1;   s_scan[bi] += s_scan[ai]; }                                                                                          \n"
    "        if (lane_id < 8)  { ai = 4 * baseai - 1;  bi = 4 * basebi - 1;   s_scan[bi] += s_scan[ai]; }                                                                                            \n"
    "        if (lane_id < 4)  { ai = 8 * baseai - 1;  bi = 8 * basebi - 1;   s_scan[bi] += s_scan[ai]; }                                                                                            \n"
    "        if (lane_id < 2)  { ai = 16 * baseai - 1;  bi = 16 * basebi - 1;   s_scan[bi] += s_scan[ai]; }                                                                                          \n"
    "        if (lane_id == 0) { s_scan[63] += s_scan[31]; s_scan[64] = s_scan[63]; s_scan[63] = 0; temp = s_scan[31]; s_scan[31] = 0; s_scan[63] += temp; }                                         \n"
    "        if (lane_id < 2)  { ai = 16 * baseai - 1;  bi = 16 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}                                                     \n"
    "        if (lane_id < 4)  { ai = 8 * baseai - 1;  bi = 8 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}                                                       \n"
    "        if (lane_id < 8)  { ai = 4 * baseai - 1;  bi = 4 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}                                                       \n"
    "        if (lane_id < 16) { ai =  2 * baseai - 1;  bi =  2 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}                                                     \n"
    "        if (lane_id < 32) { ai = baseai - 1;   bi = basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp; }                                                             \n"
    "    }                                                                                                                                                                                           \n"
    "                                                                                                                                                                                                \n"
    "    inline                                                                                                                                                                                      \n"
    "    void scan_256_plus1(__local volatile int *s_scan,                                                                                                                                           \n"
    "                  const int      lane_id)                                                                                                                                                       \n"
    "    {                                                                                                                                                                                           \n"
    "        int ai, bi;                                                                                                                                                                             \n"
    "        int baseai = 1 + 2 * lane_id;                                                                                                                                                           \n"
    "        int basebi = baseai + 1;                                                                                                                                                                \n"
    "        int temp;                                                                                                                                                                               \n"
    "                                                                                                                                                                                                \n"
    "        if (lane_id < 128) { ai = baseai - 1;     bi = basebi - 1;     s_scan[bi] += s_scan[ai]; }                                                                                              \n"
    "        barrier(CLK_LOCAL_MEM_FENCE);                                                                                                                                                           \n"
    "        if (lane_id < 64) { ai =  2 * baseai - 1;  bi =  2 * basebi - 1;   s_scan[bi] += s_scan[ai]; } barrier(CLK_LOCAL_MEM_FENCE);                                                                                         \n"
    "        if (lane_id < 32) { ai =  4 * baseai - 1;  bi =  4 * basebi - 1;   s_scan[bi] += s_scan[ai]; }                                                                                          \n"
    "        if (lane_id < 16) { ai =  8 * baseai - 1;  bi =  8 * basebi - 1;   s_scan[bi] += s_scan[ai]; }                                                                                          \n"
    "        if (lane_id < 8)  { ai = 16 * baseai - 1;  bi = 16 * basebi - 1;   s_scan[bi] += s_scan[ai]; }                                                                                          \n"
    "        if (lane_id < 4)  { ai = 32 * baseai - 1;  bi = 32 * basebi - 1;   s_scan[bi] += s_scan[ai]; }                                                                                          \n"
    "        if (lane_id < 2)  { ai = 64 * baseai - 1;  bi = 64 * basebi - 1;   s_scan[bi] += s_scan[ai]; }                                                                                          \n"
    "        if (lane_id == 0) { s_scan[255] += s_scan[127]; s_scan[256] = s_scan[255]; s_scan[255] = 0; temp = s_scan[127]; s_scan[127] = 0; s_scan[255] += temp; }                                 \n"
    "        if (lane_id < 2)  { ai = 64 * baseai - 1;  bi = 64 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}                                                     \n"
    "        if (lane_id < 4)  { ai = 32 * baseai - 1;  bi = 32 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}                                                     \n"
    "        if (lane_id < 8)  { ai = 16 * baseai - 1;  bi = 16 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}                                                     \n"
    "        if (lane_id < 16) { ai =  8 * baseai - 1;  bi =  8 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}                                                     \n"
    "        if (lane_id < 32) { ai =  4 * baseai - 1;  bi =  4 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}                                                     \n"
    "        if (lane_id < 64) { ai =  2 * baseai - 1;  bi =  2 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;} barrier(CLK_LOCAL_MEM_FENCE);                                                    \n"
    "        if (lane_id < 128) { ai = baseai - 1;   bi = basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp; }                                                            \n"
    "    }                                                                                                                                                                                           \n"
    "                                                                                                                                                                                                \n"
    "    __kernel                                                                                                                                                                                    \n"
    "    void warmup_kernel(__global int *d_scan)                                                                                                                                                    \n"
    "    {                                                                                                                                                                                           \n"
    "        volatile __local int s_scan[65];                                                                                                                                                        \n"
    "        s_scan[get_local_id(0)] = 1;                                                                                                                                                            \n"
    "        int local_id = get_local_id(0); \n"
    "        scan_64_plus1(s_scan, local_id);                                                                                                                                                \n"
    "        if(!get_group_id(0))                                                                                                                                                                    \n"
    "            d_scan[get_local_id(0)] = s_scan[get_local_id(0)];                                                                                                                                  \n"
    "    }                                                                                                                                                                                           \n"
    "                                                                                                                                                                                                \n"
    "    __kernel                                                                                                                                                                                    \n"
    "    void generate_partition_pointer_s1_kernel(__global const iT     *d_row_pointer,                                                                                                             \n"
    "                                              __global uiT          *d_partition_pointer,                                                                                                       \n"
    "                                              const int     sigma,                                                                                                                              \n"
    "                                              const iT      p,                                                                                                                                  \n"
    "                                              const iT      m,                                                                                                                                  \n"
    "                                              const iT      nnz)                                                                                                                                \n"
    "    {                                                                                                                                                                                           \n"
    "        // global thread id                                                                                                                                                                     \n"
    "        iT global_id = get_global_id(0);                                                                                                                                                        \n"
    "                                                                                                                                                                                                \n"
    "        // compute partition boundaries by partition of size sigma * omega                                                                                                                      \n"
    "        iT boundary = global_id * sigma * ANONYMOUSLIB_CSR5_OMEGA;                                                                                                                                  \n"
    "                                                                                                                                                                                                \n"
    "        // clamp partition boundaries to [0, nnz]                                                                                                                                               \n"
    "        boundary = boundary > nnz ? nnz : boundary;                                                                                                                                             \n"
    "                                                                                                                                                                                                \n"
    "        // binary search                                                                                                                                                                        \n"
    "        if (global_id <= p)                                                                                                                                                                     \n"
    "            d_partition_pointer[global_id] = binary_search_right_boundary_kernel(d_row_pointer, boundary, m + 1) - 1;                                                                           \n"
    "    }                                                                                                                                                                                           \n"
    "                                                                                                                                                                                                \n"
    "    __kernel                                                                                                                                                                                    \n"
    "    void generate_partition_pointer_s2_kernel(__global const iT   *d_row_pointer,                                                                                                               \n"
    "                                              __global uiT        *d_partition_pointer)                                                                                                         \n"
    "    {                                                                                                                                                                                           \n"
    "        const iT group_id = get_group_id(0);                                                                                                                                                    \n"
    "        const int local_id = get_local_id(0);                                                                                                                                                   \n"
    "        const iT local_size = get_local_size(0);                                                                                                                                                \n"
    "                                                                                                                                                                                                \n"
    "        volatile __local int s_dirty[1];                                                                                                                                                        \n"
    "                                                                                                                                                                                                \n"
    "        if (!local_id)                                                                                                                                                                          \n"
    "            s_dirty[0] = 0;                                                                                                                                                                     \n"
    "        barrier(CLK_LOCAL_MEM_FENCE);                                                                                                                                                           \n"
    "                                                                                                                                                                                                \n"
    "        uiT start = d_partition_pointer[group_id];                                                                                                                                              \n"
    "        uiT stop  = d_partition_pointer[group_id+1];                                                                                                                                            \n"
    "                                                                                                                                                                                                \n"
    "        start = (start << 1) >> 1;                                                                                                                                                              \n"
    "        stop  = (stop << 1) >> 1;                                                                                                                                                               \n"
    "                                                                                                                                                                                                \n"
    "        if(start == stop)                                                                                                                                                                       \n"
    "            return;                                                                                                                                                                             \n"
    "                                                                                                                                                                                                \n"
    "        uiT num_row_in_partition = stop + 1 - start;                                                                                                                                            \n"
    "        int loop = ceil((float)num_row_in_partition / (float)local_size);                                                                                                                       \n"
    "        iT row_idx, row_off_l, row_off_r;                                                                                                                                                       \n"
    "                                                                                                                                                                                                \n"
    "        for (int i = 0; i < loop; i++)                                                                                                                                                          \n"
    "        {                                                                                                                                                                                       \n"
    "            row_idx = i * local_size + start + local_id;                                                                                                                                        \n"
    "                                                                                                                                                                                                \n"
    "            if (row_idx < stop)                                                                                                                                                                 \n"
    "            {                                                                                                                                                                                   \n"
    "                row_off_l = d_row_pointer[row_idx];                                                                                                                                             \n"
    "                row_off_r = d_row_pointer[row_idx+1];                                                                                                                                           \n"
    "                                                                                                                                                                                                \n"
    "                if (row_off_l == row_off_r)                                                                                                                                                     \n"
    "                    s_dirty[0] = 1;                                                                                                                                                             \n"
    "            }                                                                                                                                                                                   \n"
    "            barrier(CLK_LOCAL_MEM_FENCE);                                                                                                                                                       \n"
    "                                                                                                                                                                                                \n"
    "            if (s_dirty[0])                                                                                                                                                                     \n"
    "                break;                                                                                                                                                                          \n"
    "        }                                                                                                                                                                                       \n"
    "                                                                                                                                                                                                \n"
    "        if (s_dirty[0] && !local_id)                                                                                                                                                            \n"
    "        {                                                                                                                                                                                       \n"
    "            start |= sizeof(uiT) == 4 ? 0x80000000 : 0x8000000000000000;                                                                                                                        \n"
    "            d_partition_pointer[group_id] = start;                                                                                                                                              \n"
    "        }                                                                                                                                                                                       \n"
    "    }                                                                                                                                                                                           \n"
    "                                                                                                                                                                                                \n"
    "    __kernel                                                                                                                                                                                    \n"
    "    void generate_partition_descriptor_s0_kernel(__global uiT         *d_partition_descriptor,                                                                                                  \n"
    "                                                 const int    num_packet)                                                                                                                       \n"
    "    {                                                                                                                                                                                           \n"
    "        const int local_id = get_local_id(0);                                                                                                                                                  \n"
    "                                                                                                                                                                                                \n"
    "        for (int i = 0; i < num_packet; i++)                                                                                                                                     \n"
    "            d_partition_descriptor[get_group_id(0) * ANONYMOUSLIB_CSR5_OMEGA * num_packet + i * ANONYMOUSLIB_CSR5_OMEGA + local_id] = 0;                                                                                                                                                                                \n"
    "    }                                                                                                                                                                                           \n"
    "       \n"
    "    __kernel                                                                                                                                                                                    \n"
    "    void generate_partition_descriptor_s1_kernel(__global const iT    *d_row_pointer,                                                                                                           \n"
    "                                                 __global uiT         *d_partition_descriptor,                                                                                                  \n"
    "                                                 const iT     m,                                                                                                                                \n"
    "                                                 const int    sigma,                                                                                                                            \n"
    "                                                 const int    bit_all_offset,                                                                                                                   \n"
    "                                                 const int    num_packet)                                                                                                                       \n"
    "    {                                                                                                                                                                                           \n"
    "        const iT global_id = get_global_id(0);                                                                                                                                                  \n"
    "                                                                                                                                                                                                \n"
    "        if (global_id < m)                                                                                                                                                                      \n"
    "        {                                                                                                                                                                                       \n"
    "            const iT row_offset = d_row_pointer[global_id];                                                                                                                                     \n"
    "                                                                                                                                                                                                \n"
    "            const iT  gx    = row_offset / sigma;                                                                                                                                               \n"
    "                                                                                                                                                                                                \n"
    "            const iT  lx    = gx % ANONYMOUSLIB_CSR5_OMEGA;                                                                                                                                         \n"
    "            const iT  pid   = gx / ANONYMOUSLIB_CSR5_OMEGA;                                                                                                                                         \n"
    "                                                                                                                                                                                                \n"
    "            const int glid  = row_offset % sigma + bit_all_offset;                                                                                                                              \n"
    "            const int llid  = glid % 32;                                                                                                                                                        \n"
    "            const int ly    = glid / 32;                                                                                                                                                        \n"
    "                                                                                                                                                                                                \n"
    "            const uiT val = 0x1 << (31 - llid);                                                                                                                                                 \n"
    "                                                                                                                                                                                                \n"
    "            const int location = pid * ANONYMOUSLIB_CSR5_OMEGA * num_packet + ly * ANONYMOUSLIB_CSR5_OMEGA + lx;                                                                                        \n"
    "                                                                                                                                                                                                \n"
    "            atomic_or(&d_partition_descriptor[location], val);                                                                                                                                  \n"
    "        }                                                                                                                                                                                       \n"
    "    }                                                                                                                                                                                           \n"
    "                                                                                                                                                                                                \n"
    "    __kernel                                                                                                                                                                                    \n"
    "    void generate_partition_descriptor_s2_kernel(__global const uiT    *d_partition_pointer,                                                                                                    \n"
    "                                                 __global uiT          *d_partition_descriptor,                                                                                                 \n"
    "                                                 __global iT           *d_partition_descriptor_offset_pointer,                                                                                  \n"
    "                                                 const int     sigma,                                                                                                                           \n"
    "                                                 const int     num_packet,                                                                                                                      \n"
    "                                                 const int     bit_y_offset,                                                                                                                    \n"
    "                                                 const int     bit_scansum_offset,                                                                                                              \n"
    "                                                 const int     p)  \n"
    "    {                                                                                                                                                                                           \n"
    "        const int local_id = get_local_id(0);                                                                                                                                                   \n"
    "                                                                                                                                                                                                \n"
    "        const int lane_id = local_id % ANONYMOUSLIB_CSR5_OMEGA;                                                                                                                                     \n"
    "        const int bunch_id = local_id / ANONYMOUSLIB_CSR5_OMEGA;                                                                                                                                    \n"
    "        const int par_id = get_global_id(0) / ANONYMOUSLIB_CSR5_OMEGA;                                                                                                                              \n"
    "                                                                                                                                                                                                \n"
    "        volatile __local uiT s_row_start_stop[ANONYMOUSLIB_THREAD_GROUP / ANONYMOUSLIB_CSR5_OMEGA + 1];                                                                                                 \n"
    "                                                                                                                                                                                                \n"
    "        if (local_id < ANONYMOUSLIB_THREAD_GROUP / ANONYMOUSLIB_CSR5_OMEGA + 1)                                                                                                                         \n"
    "            s_row_start_stop[local_id] = d_partition_pointer[par_id + local_id];                                                                                                                \n"
    "        barrier(CLK_LOCAL_MEM_FENCE);                                                                                                                                                           \n"
    "                                                                                                                                                                                                \n"
    "        uiT row_start       = s_row_start_stop[bunch_id];                                                                                                                                       \n"
    "        bool with_empty_rows = (row_start >> 31) & 0x1;                                                                                                                                         \n"
    "        row_start          &= 0x7FFFFFFF; //( << 1) >> 1                                                                                                                                        \n"
    "        const iT row_stop   = s_row_start_stop[bunch_id + 1] & 0x7FFFFFFF;                                                                                                                      \n"
    "                                                                                                                                                                                                \n"
    "        // if this is fast track partition, do not generate its partition_descriptor                                                                                                            \n"
    "        if (row_start == row_stop)                                                                                                                                                              \n"
    "        {    \n"
    "            if (!lane_id)     \n"
    "                d_partition_descriptor_offset_pointer[par_id] = 0;    \n"
    "            return;                                                                                                                                                                             \n"
    "        }                                                                                                                                                                                        \n"
    "        int y_offset = 0;                                                                                                                                                                       \n"
    "        int scansum_offset = 0;                                                                                                                                                                 \n"
    "                                                                                                                                                                                                \n"
    "        int start = 0, stop = 0, segn = 0;                                                                                                                                                      \n"
    "        bool present = 0;                                                                                                                                                                       \n"
    "        uiT bitflag = 0;                                                                                                                                                                        \n"
    "                                                                                                                                                                                                \n"
    "        volatile __local int s_segn_scan[(ANONYMOUSLIB_CSR5_OMEGA + 1) * ANONYMOUSLIB_THREAD_GROUP / ANONYMOUSLIB_CSR5_OMEGA];                                                                              \n"
    "        volatile __local int s_present[(ANONYMOUSLIB_CSR5_OMEGA + 1) * ANONYMOUSLIB_THREAD_GROUP / ANONYMOUSLIB_CSR5_OMEGA];                                                                                \n"
    "                                                                                                                                                                                                \n"
    "        const int bit_all_offset = bit_y_offset + bit_scansum_offset;                                                                                                                           \n"
    "                                                                                                                                                                                                \n"
    "        present |= !lane_id;                                                                                                                                                                    \n"
    "                                                                                                                                                                                                \n"
    "        // extract the first bit-flag packet                                                                                                                                                    \n"
    "        int ly = 0;                                                                                                                                                                             \n"
    "        uiT first_packet = d_partition_descriptor[par_id * ANONYMOUSLIB_CSR5_OMEGA * num_packet + lane_id];                                                                                         \n"
    "        bitflag = (first_packet << bit_all_offset) | ((uiT)present << 31);                                                                                                                      \n"
    "        start = !((bitflag >> 31) & 0x1);                                                                                                                                                       \n"
    "        present |= (bitflag >> 31) & 0x1;                                                                                                                                                       \n"
    "                                                                                                                                                                                                \n"
    "        #pragma unroll                                                                                                                                                                          \n"
    "        for (int i = 1; i < ANONYMOUSLIB_CSR5_SIGMA; i++)                                                                                                                                           \n"
    "        {                                                                                                                                                                                       \n"
    "            if ((!ly && i == 32 - bit_all_offset) || (ly && (i - (32 - bit_all_offset)) % 32 == 0))                                                                                             \n"
    "            {                                                                                                                                                                                   \n"
    "                ly++;                                                                                                                                                                           \n"
    "                bitflag = d_partition_descriptor[par_id * ANONYMOUSLIB_CSR5_OMEGA * num_packet + ly * ANONYMOUSLIB_CSR5_OMEGA + lane_id];                                                               \n"
    "            }                                                                                                                                                                                   \n"
    "            const int norm_i = !ly ? i : i - (32 - bit_all_offset);                                                                                                                             \n"
    "            stop += (bitflag >> (31 - norm_i % 32) ) & 0x1;                                                                                                                                     \n"
    "            present |= (bitflag >> (31 - norm_i % 32)) & 0x1;                                                                                                                                   \n"
    "        }                                                                                                                                                                                       \n"
    "                                                                                                                                                                                                \n"
    "        // compute y_offset for all partitions                                                                                                                                                  \n"
    "        segn = stop - start + present;                                                                                                                                                          \n"
    "        segn = segn > 0 ? segn : 0;                                                                                                                                                             \n"
    "                                                                                                                                                                                                \n"
    "        s_segn_scan[bunch_id * (ANONYMOUSLIB_CSR5_OMEGA + 1) + lane_id] = segn;                                                                                                                     \n"
    "        scan_64_plus1(&s_segn_scan[bunch_id * (ANONYMOUSLIB_CSR5_OMEGA + 1)], lane_id); // exclusive scan                                                                                           \n"
    "        if (!lane_id && !with_empty_rows)     \n"
    "            d_partition_descriptor_offset_pointer[par_id] = 0;    \n"
    "        if (!lane_id && with_empty_rows)                                                                                                                                                        \n"
    "        {   \n"
    "            d_partition_descriptor_offset_pointer[par_id] = s_segn_scan[bunch_id * (ANONYMOUSLIB_CSR5_OMEGA + 1) + ANONYMOUSLIB_CSR5_OMEGA]; // the total number of segments in this partition          \n"
    "            //d_partition_descriptor_offset_pointer[p] = 1;\n"
    "        }   \n"
    "        y_offset = s_segn_scan[bunch_id * (ANONYMOUSLIB_CSR5_OMEGA + 1) + lane_id];                                                                                                                 \n"
    "                                                                                                                                                                                                \n"
    "        // compute scansum_offset                                                                                                                                                               \n"
    "        s_present[bunch_id * (ANONYMOUSLIB_CSR5_OMEGA + 1) + lane_id] = present;                                                                                                                    \n"
    "        int next1 = lane_id + 1;                                                                                                                                                                \n"
    "        if (present)                                                                                                                                                                            \n"
    "        {                                                                                                                                                                                       \n"
    "            while (!s_present[bunch_id * (ANONYMOUSLIB_CSR5_OMEGA + 1) + next1] && next1 < ANONYMOUSLIB_CSR5_OMEGA)                                                                                     \n"
    "            {                                                                                                                                                                                   \n"
    "                scansum_offset++;                                                                                                                                                               \n"
    "                next1++;                                                                                                                                                                        \n"
    "            }                                                                                                                                                                                   \n"
    "        }                                                                                                                                                                                       \n"
    "                                                                                                                                                                                                \n"
    "        y_offset = lane_id ? y_offset - 1 : 0;                                                                                                                                                  \n"
    "                                                                                                                                                                                                \n"
    "        first_packet |= y_offset << (32 - bit_y_offset);                                                                                                                                        \n"
    "        first_packet |= scansum_offset << (32 - bit_all_offset);                                                                                                                                \n"
    "                                                                                                                                                                                                \n"
    "        d_partition_descriptor[par_id * ANONYMOUSLIB_CSR5_OMEGA * num_packet + lane_id] = first_packet;                                                                                             \n"
    "    }                                                                                                                                                                                           \n"
    "                                                                                                                                                                                                \n"
    "    __kernel                                                                                                                                                                                    \n"
    "    void generate_partition_descriptor_s3_kernel(__global iT           *d_partition_descriptor_offset_pointer,                                                                                  \n"
    "                                                 const int     p)                                                                                                                               \n"
    "    {                                                                                                                                                                                           \n"
    "        const int local_id = get_local_id(0);                                                                                                                                                   \n"
    "        const int local_size = get_local_size(0);                                                                                                                                               \n"
    "                                                                                                                                                                                                \n"
    "        iT sum = 0;                                                                                                                                                                             \n"
    "        volatile __local iT s_partition_descriptor_offset_pointer[256 + 1];                                                                                                                     \n"
    "                                                                                                                                                                                                \n"
    "        int loop = ceil((float)p / (float)local_size);                                                                                                                                          \n"
    "                                                                                                                                                                                                \n"
    "        for (int i = 0; i < loop; i++)                                                                                                                                                          \n"
    "        {                                                                                                                                                                                       \n"
    "            s_partition_descriptor_offset_pointer[local_id] = (local_id + i * local_size < p) ? d_partition_descriptor_offset_pointer[local_id + i * local_size] : 0;                           \n"
    "            barrier(CLK_LOCAL_MEM_FENCE);                                                                                                                                                       \n"
    "                                                                                                                                                                                                \n"
    "            scan_256_plus1(s_partition_descriptor_offset_pointer, local_id);                                                                                                                    \n"
    "            barrier(CLK_LOCAL_MEM_FENCE);                                                                                                                                                       \n"
    "                                                                                                                                                                                                \n"
    "            s_partition_descriptor_offset_pointer[local_id] += sum;                                                                                                                             \n"
    "            if (!local_id)                                                                                                                                                                      \n"
    "                s_partition_descriptor_offset_pointer[256] += sum;                                                                                                                              \n"
    "            barrier(CLK_LOCAL_MEM_FENCE);                                                                                                                                                       \n"
    "                                                                                                                                                                                                \n"
    "            sum = s_partition_descriptor_offset_pointer[256];                                                                                                                                   \n"
    "                                                                                                                                                                                                \n"
    "            if (local_id + i * local_size < p + 1)                                                                                                                                              \n"
    "                d_partition_descriptor_offset_pointer[local_id + i * local_size] = s_partition_descriptor_offset_pointer[local_id];                                                             \n"
    "        }                                                                                                                                                                                       \n"
    "    }                                                                                                                                                                                           \n"
    "                                                                                                                                                                                                \n"
    "    inline                                                                                                                                                                                      \n"
    "    void partition_normal_track_empty_Ologn(__global const iT             *d_row_pointer,                                                                                                       \n"
    "                                            __global const uiT            *d_partition_descriptor,                                                                                              \n"
    "                                            __global const iT             *d_partition_descriptor_offset_pointer,                                                                               \n"
    "                                            __global iT                   *d_partition_descriptor_offset,                                                                                       \n"
    "                                            const iT              par_id,                                                                                                                       \n"
    "                                            const int             lane_id,                                                                                                                      \n"
    "                                            const int             bit_y_offset,                                                                                                                 \n"
    "                                            const int             bit_scansum_offset,                                                                                                           \n"
    "                                            iT                    row_start,                                                                                                                    \n"
    "                                            const iT              row_stop,                                                                                                                     \n"
    "                                            const int             c_sigma)                                                                                                                      \n"
    "    {                                                                                                                                                                                           \n"
    "        bool local_bit;                                                                                                                                                                         \n"
    "                                                                                                                                                                                                \n"
    "        int offset_pointer = d_partition_descriptor_offset_pointer[par_id];                                                                                                                     \n"
    "                                                                                                                                                                                                \n"
    "        uiT descriptor = d_partition_descriptor[lane_id];                                                                                                                                       \n"
    "                                                                                                                                                                                                \n"
    "        int y_offset = descriptor >> (32 - bit_y_offset);                                                                                                                                       \n"
    "        const int bit_bitflag = 32 - bit_y_offset - bit_scansum_offset;                                                                                                                         \n"
    "                                                                                                                                                                                                \n"
    "        // step 1. thread-level seg sum                                                                                                                                                         \n"
    "        // extract the first bit-flag packet                                                                                                                                                    \n"
    "        int ly = 0;                                                                                                                                                                             \n"
    "        descriptor = descriptor << (bit_y_offset + bit_scansum_offset);                                                                                                                         \n"
    "        descriptor = lane_id ? descriptor : descriptor | 0x80000000;                                                                                                                            \n"
    "                                                                                                                                                                                                \n"
    "        local_bit = (descriptor >> 31) & 0x1;                                                                                                                                                   \n"
    "                                                                                                                                                                                                \n"
    "        if (local_bit && lane_id)                                                                                                                                                               \n"
    "        {                                                                                                                                                                                       \n"
    "            const iT idx = par_id * ANONYMOUSLIB_CSR5_OMEGA * c_sigma + lane_id * c_sigma;                                                                                                          \n"
    "            const iT y_index = binary_search_right_boundary_kernel(&d_row_pointer[row_start+1], idx, row_stop - row_start) - 1;                                                                 \n"
    "            d_partition_descriptor_offset[offset_pointer + y_offset] = y_index;                                                                                                                 \n"
    "                                                                                                                                                                                                \n"
    "            y_offset++;                                                                                                                                                                         \n"
    "        }                                                                                                                                                                                       \n"
    "                                                                                                                                                                                                \n"
    "        #pragma unroll                                                                                                                                                                          \n"
    "        for (int i = 1; i < ANONYMOUSLIB_CSR5_SIGMA; i++)                                                                                                                                           \n"
    "        {                                                                                                                                                                                       \n"
    "            if ((!ly && i == bit_bitflag) || (ly && !(31 & (i - bit_bitflag))))                                                                                                                 \n"
    "            {                                                                                                                                                                                   \n"
    "                ly++;                                                                                                                                                                           \n"
    "                descriptor = d_partition_descriptor[ly * ANONYMOUSLIB_CSR5_OMEGA + lane_id];                                                                                                        \n"
    "            }                                                                                                                                                                                   \n"
    "            const int norm_i = 31 & (!ly ? i : i - bit_bitflag);                                                                                                                                \n"
    "                                                                                                                                                                                                \n"
    "            local_bit = (descriptor >> (31 - norm_i)) & 0x1;                                                                                                                                    \n"
    "                                                                                                                                                                                                \n"
    "            if (local_bit)                                                                                                                                                                      \n"
    "            {                                                                                                                                                                                   \n"
    "                const iT idx = par_id * ANONYMOUSLIB_CSR5_OMEGA * c_sigma + lane_id * c_sigma + i;                                                                                                  \n"
    "                const iT y_index = binary_search_right_boundary_kernel(&d_row_pointer[row_start+1], idx, row_stop - row_start) - 1;                                                             \n"
    "                d_partition_descriptor_offset[offset_pointer + y_offset] = y_index;                                                                                                             \n"
    "                                                                                                                                                                                                \n"
    "                y_offset++;                                                                                                                                                                     \n"
    "            }                                                                                                                                                                                   \n"
    "        }                                                                                                                                                                                       \n"
    "    }                                                                                                                                                                                           \n"
    "                                                                                                                                                                                                \n"
    "    inline                                                                                                                                                                                      \n"
    "    void generate_partition_descriptor_offset_partition(__global const iT           *d_row_pointer,                                                                                             \n"
    "                                                        __global const uiT          *d_partition_pointer,                                                                                       \n"
    "                                                        __global const uiT          *d_partition_descriptor,                                                                                    \n"
    "                                                        __global const iT           *d_partition_descriptor_offset_pointer,                                                                     \n"
    "                                                        __global iT                 *d_partition_descriptor_offset,                                                                             \n"
    "                                                        const iT            par_id,                                                                                                             \n"
    "                                                        const int           lane_id,                                                                                                            \n"
    "                                                        const int           bunch_id,                                                                                                           \n"
    "                                                        const int           bit_y_offset,                                                                                                       \n"
    "                                                        const int           bit_scansum_offset,                                                                                                 \n"
    "                                                        const int           c_sigma,                                                                                                           \n"
    "                                                        __local uiT *s_row_start_stop)    \n"
    "    {                                                                                                                                                                                           \n"
    "        const int local_id    = get_local_id(0);                                                                                                                                                \n"
    "        uiT row_start, row_stop;                                                                                                                                                                \n"
    "                                                                                                                                                                                                \n"
    "        //volatile __local uiT s_row_start_stop[ANONYMOUSLIB_THREAD_GROUP / ANONYMOUSLIB_CSR5_OMEGA + 1];                                                                                                 \n"
    "        if (local_id < ANONYMOUSLIB_THREAD_GROUP / ANONYMOUSLIB_CSR5_OMEGA + 1)                                                                                                                         \n"
    "            s_row_start_stop[local_id] = d_partition_pointer[par_id + local_id];                                                                                                                \n"
    "        barrier(CLK_LOCAL_MEM_FENCE);                                                                                                                                                           \n"
    "                                                                                                                                                                                                \n"
    "        row_start = s_row_start_stop[bunch_id];                                                                                                                                                 \n"
    "        row_stop  = s_row_start_stop[bunch_id + 1] & 0x7FFFFFFF;                                                                                                                                \n"
    "                                                                                                                                                                                                \n"
    "        if (row_start >> 31) // with empty rows                                                                                                                                                 \n"
    "        {                                                                                                                                                                                       \n"
    "            row_start &= 0x7FFFFFFF;     //( << 1) >> 1                                                                                                                                         \n"
    "                                                                                                                                                                                                \n"
    "            partition_normal_track_empty_Ologn                                                                                                                                                  \n"
    "                    (d_row_pointer,                                                                                                                                                             \n"
    "                     d_partition_descriptor, d_partition_descriptor_offset_pointer, d_partition_descriptor_offset,                                                                              \n"
    "                     par_id, lane_id,                                                                                                                                                           \n"
    "                     bit_y_offset, bit_scansum_offset, row_start, row_stop, c_sigma);                                                                                                           \n"
    "        }                                                                                                                                                                                       \n"
    "        else // without empty rows                                                                                                                                                              \n"
    "        {                                                                                                                                                                                       \n"
    "            return;                                                                                                                                                                             \n"
    "        }                                                                                                                                                                                       \n"
    "    }                                                                                                                                                                                           \n"
    "                                                                                                                                                                                                \n"
    "    __kernel                                                                                                                                                                                    \n"
    "    void generate_partition_descriptor_offset_kernel(__global const iT           *d_row_pointer,                                                                                                \n"
    "                                                     __global const uiT          *d_partition_pointer,                                                                                          \n"
    "                                                     __global const uiT          *d_partition_descriptor,                                                                                       \n"
    "                                                     __global const iT           *d_partition_descriptor_offset_pointer,                                                                        \n"
    "                                                     __global iT                 *d_partition_descriptor_offset,                                                                                \n"
    "                                                     const iT            p,                                                                                                                     \n"
    "                                                     const int           num_packet,                                                                                                            \n"
    "                                                     const int           bit_y_offset,                                                                                                          \n"
    "                                                     const int           bit_scansum_offset,                                                                                                    \n"
    "                                                     const int           c_sigma)                                                                                                               \n"
    "    {                                                                                                                                                                                           \n"
    "        const int local_id    = get_local_id(0);                                                                                                                                                \n"
    "                                                                                                                                                                                                \n"
    "        // warp lane id                                                                                                                                                                         \n"
    "        const int lane_id = local_id % ANONYMOUSLIB_CSR5_OMEGA;                                                                                                                 \n"
    "        // warp global id == par_id                                                                                                                                                             \n"
    "        const iT  par_id =  get_global_id(0) / ANONYMOUSLIB_CSR5_OMEGA;                                                                                                                             \n"
    "        const int bunch_id = local_id / ANONYMOUSLIB_CSR5_OMEGA;                                                                                                                                    \n"
    "        volatile __local uiT s_row_start_stop[ANONYMOUSLIB_THREAD_GROUP / ANONYMOUSLIB_CSR5_OMEGA + 1];                                                                                    \n"
    "        if (par_id >= p - 1)                                                                                                                                                                    \n"
    "            return;                                                                                                                                                                             \n"
    "                                                                                                                                                                                                \n"
    "        generate_partition_descriptor_offset_partition                                                                                                                                          \n"
    "                    (d_row_pointer, d_partition_pointer,                                                                                                                                        \n"
    "                     &d_partition_descriptor[par_id * ANONYMOUSLIB_CSR5_OMEGA * num_packet],                                                                                                        \n"
    "                     d_partition_descriptor_offset_pointer, d_partition_descriptor_offset,                                                                                                      \n"
    "                     par_id, lane_id, bunch_id, bit_y_offset, bit_scansum_offset, c_sigma, s_row_start_stop);                                                                                                     \n"
    "    }                                                                                                                                                                                           \n"
    "                                                                                                                                                                                                \n"
    "    __kernel                                                                                                                                                                                    \n"
    "    void aosoa_transpose_kernel_smem_iT(__global iT         *d_data,                                                                                                                            \n"
    "                                     __global const uiT *d_partition_pointer,                                                                                                                   \n"
    "                                     const int R2C) // R2C==true means CSR->CSR5, otherwise CSR5->CSR                                                                                          \n"
    "    {                                                                                                                                                                                           \n"
    "        __local uiT s_par[2];                                                                                                                                                                   \n"
    "                                                                                                                                                                                                \n"
    "        const int local_id = get_local_id(0);                                                                                                                                                   \n"
    "                                                                                                                                                                                                \n"
    "        if (local_id < 2)                                                                                                                                                                       \n"
    "            s_par[local_id] = d_partition_pointer[get_group_id(0) + local_id];                                                                                                                  \n"
    "        barrier(CLK_LOCAL_MEM_FENCE);                                                                                                                                                           \n"
    "                                                                                                                                                                                                \n"
    "        // if this is fast track partition, do not transpose it                                                                                                                                 \n"
    "        if (s_par[0] == s_par[1])                                                                                                                                                               \n"
    "            return;                                                                                                                                                                             \n"
    "                                                                                                                                                                                                \n"
    "        __local iT s_data[ANONYMOUSLIB_CSR5_SIGMA * (ANONYMOUSLIB_CSR5_OMEGA + 1)];                                                                                                                     \n"
    "                                                                                                                                                                                                \n"
    "        // load global data to shared mem                                                                                                                                                       \n"
    "        int idx_y, idx_x;                                                                                                                                                                       \n"
    "        for (int idx = local_id; idx < ANONYMOUSLIB_CSR5_OMEGA * ANONYMOUSLIB_CSR5_SIGMA; idx += get_local_size(0))                                                                                     \n"
    "        {                                                                                                                                                                                       \n"
    "            if (R2C)                                                                                                                                                                            \n"
    "            {                                                                                                                                                                                   \n"
    "                idx_y = idx % ANONYMOUSLIB_CSR5_SIGMA;                                                                                                                                              \n"
    "                idx_x = idx / ANONYMOUSLIB_CSR5_SIGMA;                                                                                                                                              \n"
    "            }                                                                                                                                                                                   \n"
    "            else                                                                                                                                                                                \n"
    "            {                                                                                                                                                                                   \n"
    "                idx_x = idx % ANONYMOUSLIB_CSR5_OMEGA;                                                                                                                                              \n"
    "                idx_y = idx / ANONYMOUSLIB_CSR5_OMEGA;                                                                                                                                              \n"
    "            }                                                                                                                                                                                   \n"
    "                                                                                                                                                                                                \n"
    "            s_data[idx_y * (ANONYMOUSLIB_CSR5_OMEGA+1) + idx_x] = d_data[get_group_id(0) * ANONYMOUSLIB_CSR5_OMEGA * ANONYMOUSLIB_CSR5_SIGMA + idx];                                                        \n"
    "        }                                                                                                                                                                                       \n"
    "        barrier(CLK_LOCAL_MEM_FENCE);                                                                                                                                                           \n"
    "                                                                                                                                                                                                \n"
    "        // store transposed shared mem data to global                                                                                                                                           \n"
    "        for (int idx = local_id; idx < ANONYMOUSLIB_CSR5_OMEGA * ANONYMOUSLIB_CSR5_SIGMA; idx += get_local_size(0))                                                                                     \n"
    "        {                                                                                                                                                                                       \n"
    "            if (R2C)                                                                                                                                                                            \n"
    "            {                                                                                                                                                                                   \n"
    "                idx_x = idx % ANONYMOUSLIB_CSR5_OMEGA;                                                                                                                                              \n"
    "                idx_y = idx / ANONYMOUSLIB_CSR5_OMEGA;                                                                                                                                              \n"
    "            }                                                                                                                                                                                   \n"
    "            else                                                                                                                                                                                \n"
    "            {                                                                                                                                                                                   \n"
    "                idx_y = idx % ANONYMOUSLIB_CSR5_SIGMA;                                                                                                                                              \n"
    "                idx_x = idx / ANONYMOUSLIB_CSR5_SIGMA;                                                                                                                                              \n"
    "            }                                                                                                                                                                                   \n"
    "                                                                                                                                                                                                \n"
    "            d_data[get_group_id(0) * ANONYMOUSLIB_CSR5_OMEGA * ANONYMOUSLIB_CSR5_SIGMA + idx] = s_data[idx_y * (ANONYMOUSLIB_CSR5_OMEGA+1) + idx_x];                                                        \n"
    "        }                                                                                                                                                                                       \n"
    "    }                                                                                                                                                                                           \n"
    "                                                                                                                                                                                                \n"
    "    __kernel                                                                                                                                                                                    \n"
    "    void aosoa_transpose_kernel_smem_vT(__global vT         *d_data,                                                                                                                            \n"
    "                                     __global const uiT *d_partition_pointer,                                                                                                                   \n"
    "                                     const int R2C) // R2C==true means CSR->CSR5, otherwise CSR5->CSR                                                                                          \n"
    "    {                                                                                                                                                                                           \n"
    "        __local uiT s_par[2];                                                                                                                                                                   \n"
    "                                                                                                                                                                                                \n"
    "        const int local_id = get_local_id(0);                                                                                                                                                   \n"
    "                                                                                                                                                                                                \n"
    "        if (local_id < 2)                                                                                                                                                                       \n"
    "            s_par[local_id] = d_partition_pointer[get_group_id(0) + local_id];                                                                                                                  \n"
    "        barrier(CLK_LOCAL_MEM_FENCE);                                                                                                                                                           \n"
    "                                                                                                                                                                                                \n"
    "        // if this is fast track partition, do not transpose it                                                                                                                                 \n"
    "        if (s_par[0] == s_par[1])                                                                                                                                                               \n"
    "            return;                                                                                                                                                                             \n"
    "                                                                                                                                                                                                \n"
    "        __local vT s_data[ANONYMOUSLIB_CSR5_SIGMA * (ANONYMOUSLIB_CSR5_OMEGA + 1)];                                                                                                                     \n"
    "                                                                                                                                                                                                \n"
    "        // load global data to shared mem                                                                                                                                                       \n"
    "        int idx_y, idx_x;                                                                                                                                                                       \n"
    "        for (int idx = local_id; idx < ANONYMOUSLIB_CSR5_OMEGA * ANONYMOUSLIB_CSR5_SIGMA; idx += get_local_size(0))                                                                                     \n"
    "        {                                                                                                                                                                                       \n"
    "            if (R2C)                                                                                                                                                                            \n"
    "            {                                                                                                                                                                                   \n"
    "                idx_y = idx % ANONYMOUSLIB_CSR5_SIGMA;                                                                                                                                              \n"
    "                idx_x = idx / ANONYMOUSLIB_CSR5_SIGMA;                                                                                                                                              \n"
    "            }                                                                                                                                                                                   \n"
    "            else                                                                                                                                                                                \n"
    "            {                                                                                                                                                                                   \n"
    "                idx_x = idx % ANONYMOUSLIB_CSR5_OMEGA;                                                                                                                                              \n"
    "                idx_y = idx / ANONYMOUSLIB_CSR5_OMEGA;                                                                                                                                              \n"
    "            }                                                                                                                                                                                   \n"
    "                                                                                                                                                                                                \n"
    "            s_data[idx_y * (ANONYMOUSLIB_CSR5_OMEGA+1) + idx_x] = d_data[get_group_id(0) * ANONYMOUSLIB_CSR5_OMEGA * ANONYMOUSLIB_CSR5_SIGMA + idx];                                                        \n"
    "        }                                                                                                                                                                                       \n"
    "        barrier(CLK_LOCAL_MEM_FENCE);                                                                                                                                                           \n"
    "                                                                                                                                                                                                \n"
    "        // store transposed shared mem data to global                                                                                                                                           \n"
    "        for (int idx = local_id; idx < ANONYMOUSLIB_CSR5_OMEGA * ANONYMOUSLIB_CSR5_SIGMA; idx += get_local_size(0))                                                                                     \n"
    "        {                                                                                                                                                                                       \n"
    "            if (R2C)                                                                                                                                                                            \n"
    "            {                                                                                                                                                                                   \n"
    "                idx_x = idx % ANONYMOUSLIB_CSR5_OMEGA;                                                                                                                                              \n"
    "                idx_y = idx / ANONYMOUSLIB_CSR5_OMEGA;                                                                                                                                              \n"
    "            }                                                                                                                                                                                   \n"
    "            else                                                                                                                                                                                \n"
    "            {                                                                                                                                                                                   \n"
    "                idx_y = idx % ANONYMOUSLIB_CSR5_SIGMA;                                                                                                                                              \n"
    "                idx_x = idx / ANONYMOUSLIB_CSR5_SIGMA;                                                                                                                                              \n"
    "            }                                                                                                                                                                                   \n"
    "                                                                                                                                                                                                \n"
    "            d_data[get_group_id(0) * ANONYMOUSLIB_CSR5_OMEGA * ANONYMOUSLIB_CSR5_SIGMA + idx] = s_data[idx_y * (ANONYMOUSLIB_CSR5_OMEGA+1) + idx_x];                                                        \n"
    "        }                                                                                                                                                                                       \n"
    "    }                                                                                                                                                                                           \n"
    "    \n";

    _ocl_source_code_string_csr5_spmv_const =
    "    #pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable                                                                             \n"
    "    #pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable                                                                         \n"
    "    #pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable                                                                                    \n"
    "    #pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable                                                                                \n"
    "                                                                                                                                                   \n"
    "    #pragma OPENCL EXTENSION cl_khr_fp64 : enable                                                                                                  \n"
    "                                                                                                                                                   \n"
    "    #define  ANONYMOUSLIB_CSR5_OMEGA _REPLACE_ANONYMOUSLIB_CSR5_OMEGA_SEGMENT_ \n"
    "    #define  ANONYMOUSLIB_CSR5_SIGMA _REPLACE_ANONYMOUSLIB_CSR5_SIGMA_SEGMENT_ \n"
    "    #define  ANONYMOUSLIB_THREAD_GROUP _REPLACE_ANONYMOUSLIB_THREAD_GROUP_SEGMENT_ \n"
    "    #define  ANONYMOUSLIB_THREAD_BUNCH _REPLACE_ANONYMOUSLIB_THREAD_BUNCH_SEGMENT_ \n"
    "    typedef _REPLACE_ANONYMOUSLIB_CSR5_INDEX_TYPE_SEGMENT_   iT; \n"
    "    typedef _REPLACE_ANONYMOUSLIB_CSR5_UNSIGNED_INDEX_TYPE_SEGMENT_   uiT; \n"
    "    typedef _REPLACE_ANONYMOUSLIB_CSR5_VALUE_TYPE_SEGMENT_   vT; \n"
    "                                                                                                                                                   \n"
    "    inline                                                                                                                                         \n"
    "    void sum_64(__local volatile  vT *s_sum,                                                                                                       \n"
    "                const int    local_id)                                                                                                             \n"
    "    {                                                                                                                                              \n"
    "        //s_sum[local_id] += s_sum[local_id + 32];                                                                                                   \n"
    "        //s_sum[local_id] += s_sum[local_id + 16];                                                                                                   \n"
    "        //s_sum[local_id] += s_sum[local_id + 8];                                                                                                    \n"
    "        //s_sum[local_id] += s_sum[local_id + 4];                                                                                                    \n"
    "        //s_sum[local_id] += s_sum[local_id + 2];                                                                                                    \n"
    "        //s_sum[local_id] += s_sum[local_id + 1];                                                                                                    \n"
    "        vT sum = s_sum[local_id];  \n"
    "        if (local_id < 16) s_sum[local_id] = sum = sum +  s_sum[local_id + 16] + s_sum[local_id + 32] + s_sum[local_id + 48]; \n"
    "        if (local_id < 4)  s_sum[local_id] = sum = sum +  s_sum[local_id + 4] + s_sum[local_id + 8] + s_sum[local_id + 12]; \n"
    "        if (local_id < 1)  s_sum[local_id] = sum = sum +  s_sum[local_id + 1] + s_sum[local_id + 2] + s_sum[local_id + 3]; \n"
    "    }                                                                                                                                              \n"
    "                                                                                                                                                   \n"
    "    inline                                                                                                                                         \n"
    "    void sum_256(__local volatile  vT *s_sum,                                                                                                      \n"
    "                const int    local_id)                                                                                                             \n"
    "    {                                                                                                                                              \n"
    "        if (local_id < 128) s_sum[local_id] += s_sum[local_id + 128];                                                                              \n"
    "        barrier(CLK_LOCAL_MEM_FENCE);                                                                                                              \n"
    "        if (local_id < 64)                                                                                                                         \n"
    "        {                                                                                                                                          \n"
    "            s_sum[local_id] += s_sum[local_id + 64];     barrier(CLK_LOCAL_MEM_FENCE);                                                                                          \n"
    "            s_sum[local_id] += s_sum[local_id + 32];     barrier(CLK_LOCAL_MEM_FENCE);                                                                                          \n"
    "            s_sum[local_id] += s_sum[local_id + 16];     barrier(CLK_LOCAL_MEM_FENCE);                                                                                          \n"
    "            s_sum[local_id] += s_sum[local_id + 8];                                                                                                \n"
    "            s_sum[local_id] += s_sum[local_id + 4];                                                                                                \n"
    "            s_sum[local_id] += s_sum[local_id + 2];                                                                                                \n"
    "            s_sum[local_id] += s_sum[local_id + 1];                                                                                                \n"
    "        }                                                                                                                                          \n"
    "    }                                                                                                                                              \n"
    "                                                                                                                                                   \n"
    "    inline                                                                                                                                         \n"
    "    void scan_64(__local volatile vT *s_scan,                                                                                                     \n"
    "                       const int      lane_id)                                                                                                     \n"
    "    {                                                                                                                                              \n"
    "        int ai, bi;                                                                                                                                \n"
    "        int baseai = 1 + 2 * lane_id;                                                                                                              \n"
    "        int basebi = baseai + 1;                                                                                                                   \n"
    "        vT temp;                                                                                                                                  \n"
    "                                                                                                                                                   \n"
    "        if (lane_id < 32) { ai = baseai - 1;     bi = basebi - 1;     s_scan[bi] += s_scan[ai]; }                                                  \n"
    "        if (lane_id < 16) { ai =  2 * baseai - 1;  bi =  2 * basebi - 1;   s_scan[bi] += s_scan[ai]; }                                             \n"
    "        if (lane_id < 8)  { ai = 4 * baseai - 1;  bi = 4 * basebi - 1;   s_scan[bi] += s_scan[ai]; }                                               \n"
    "        if (lane_id < 4)  { ai = 8 * baseai - 1;  bi = 8 * basebi - 1;   s_scan[bi] += s_scan[ai]; }                                               \n"
    "        if (lane_id < 2)  { ai = 16 * baseai - 1;  bi = 16 * basebi - 1;   s_scan[bi] += s_scan[ai]; }                                             \n"
    "        if (lane_id == 0) { s_scan[63] += s_scan[31]; s_scan[63] = 0; temp = s_scan[31]; s_scan[31] = 0; s_scan[63] += temp; }                     \n"
    "        if (lane_id < 2)  { ai = 16 * baseai - 1;  bi = 16 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}        \n"
    "        if (lane_id < 4)  { ai = 8 * baseai - 1;  bi = 8 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}          \n"
    "        if (lane_id < 8)  { ai = 4 * baseai - 1;  bi = 4 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}          \n"
    "        if (lane_id < 16) { ai =  2 * baseai - 1;  bi =  2 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}        \n"
    "        if (lane_id < 32) { ai = baseai - 1;   bi = basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp; }                \n"
    "    }                                                                                                                                              \n"
    "                                                                                                                                                   \n"
    "    inline                                                                                                                                         \n"
    "    void atom_add_fp32(volatile __global float *val,                                                                                               \n"
    "                       float delta)                                                                                                                \n"
    "    {                                                                                                                                              \n"
    "        union { float f; unsigned int i; } old;                                                                                                    \n"
    "        union { float f; unsigned int i; } new;                                                                                                    \n"
    "        do                                                                                                                                         \n"
    "        {                                                                                                                                          \n"
    "            old.f = *val;                                                                                                                          \n"
    "            new.f = old.f + delta;                                                                                                                 \n"
    "        }                                                                                                                                          \n"
    "        while (atomic_cmpxchg((volatile __global unsigned int *)val, old.i, new.i) != old.i);                                                      \n"
    "    }                                                                                                                                              \n"
    "                                                                                                                                                   \n"
    "    inline                                                                                                                                         \n"
    "    void atom_add_fp64(volatile __global double *val,                                                                                              \n"
    "                       double delta)                                                                                                               \n"
    "    {                                                                                                                                              \n"
    "        union { double f; ulong i; } old;                                                                                                          \n"
    "        union { double f; ulong i; } new;                                                                                                          \n"
    "        do                                                                                                                                       \n"
    "        {                                                                                                                                        \n"
    "            old.f = *val;                                                                                                                        \n"
    "            new.f = old.f + delta;                                                                                                               \n"
    "        }                                                                                                                                        \n"
    "        while (atom_cmpxchg((volatile __global ulong *)val, old.i, new.i) != old.i);                                                             \n"
    "    }                                                                                                                                              \n"
    "                                                                                                                                                   \n"
    "    inline                                                                                                                                         \n"
    "    vT candidate(__global const vT           *d_value_partition,                                                                                   \n"
    "                 __global const vT           *d_x,                                                                                                 \n"
    "                 __global const iT           *d_column_index_partition,                                                                            \n"
    "                 const iT            candidate_index,                                                                                              \n"
    "                 const vT            alpha)                                                                                                        \n"
    "    {                                                                                                                                              \n"
    "        vT x = d_x[d_column_index_partition[candidate_index]];                                                                                        \n"
    "        return d_value_partition[candidate_index] * x;// * alpha;                                                                                  \n"
    "    }                                                                                                                                              \n"
    "                                                                                                                                                   \n"
    "    inline                                                                                                                                         \n"
    "    vT segmented_sum(vT             tmp_sum,                                                                                                       \n"
    "                     __local volatile vT   *s_sum,                                                                                                 \n"
    "                     const int      scansum_offset,                                                                                                \n"
    "                     const int      lane_id)                                                                                                       \n"
    "    {                                                                                                                                              \n"
    "        if (lane_id)                                                                                                                               \n"
    "            s_sum[lane_id - 1] = tmp_sum;                                                                                                          \n"
    "        s_sum[lane_id] = lane_id == ANONYMOUSLIB_CSR5_OMEGA - 1 ? 0 : s_sum[lane_id];                                                                  \n"
    "        vT sum = tmp_sum = s_sum[lane_id];                                                                                                         \n"
    "        scan_64(s_sum, lane_id); // exclusive scan                                                                                                 \n"
    "        s_sum[lane_id] += tmp_sum; // inclusive scan = exclusive scan + original value                                                             \n"
    "        tmp_sum = s_sum[lane_id + scansum_offset];                                                                                                 \n"
    "        tmp_sum = tmp_sum - s_sum[lane_id] + sum;                                                                                                  \n"
    "                                                                                                                                                   \n"
    "        return tmp_sum;                                                                                                                            \n"
    "    }                                                                                                                                              \n"
    "                                                                                                                                                   \n"
    "    inline                                                                                                                                         \n"
    "    void partition_fast_track(__global const vT           *d_value_partition,                                                                      \n"
    "                              __global const vT           *d_x,                                                                                    \n"
    "                              __global const iT           *d_column_index_partition,                                                               \n"
    "                              __global vT                 *d_calibrator,                                                                           \n"
    "                              __local volatile vT        *s_sum,                                                                                   \n"
    "                              const int           lane_id,                                                                                         \n"
    "                              const iT            par_id,                                                                                          \n"
    "                              const vT            alpha)                                                                                           \n"
    "    {                                                                                                                                              \n"
    "        vT sum = 0;                                                                                                                                \n"
    "                                                                                                                                                   \n"
    "        #pragma unroll                                                                                                                             \n"
    "        for (int i = 0; i < ANONYMOUSLIB_CSR5_SIGMA; i++)                                                                                              \n"
    "            sum += candidate(d_value_partition, d_x, d_column_index_partition, i * ANONYMOUSLIB_CSR5_OMEGA + lane_id, alpha);                          \n"
    "                                                                                                                                                   \n"
    "        s_sum[lane_id] = sum;                                                                                                                      \n"
    "        sum_64(s_sum, lane_id);                                                                                                                    \n"
    "        if (!lane_id)                                                                                                                              \n"
    "            d_calibrator[par_id] = s_sum[0];                                                                                                       \n"
    "    }                                                                                                                                              \n"
    "                                                                                                                                                   \n"
    "    inline                                                                                                                                         \n"
    "    void partition_normal_track(__global const iT           *d_column_index_partition,                                                             \n"
    "                                __global const vT           *d_value_partition,                                                                    \n"
    "                                __global const vT           *d_x,                                                                                  \n"
    "                                __global const uiT          *d_partition_descriptor,                                                               \n"
    "                                __global const iT           *d_partition_descriptor_offset_pointer,                                                \n"
    "                                __global const iT           *d_partition_descriptor_offset,                                                        \n"
    "                                __global vT                 *d_calibrator,                                                                         \n"
    "                                __global vT                 *d_y,                                                                                  \n"
    "                                __local volatile vT         *s_sum,                                                                                \n"
    "                                __local volatile int        *s_scan,                                                                               \n"
    "                                const iT            par_id,                                                                                        \n"
    "                                const int           lane_id,                                                                                       \n"
    "                                const int           bit_y_offset,                                                                                  \n"
    "                                const int           bit_scansum_offset,                                                                            \n"
    "                                iT                  row_start,                                                                                     \n"
    "                                const bool          empty_rows,                                                                                    \n"
    "                                const vT            alpha)                                                                                         \n"
    "    {                                                                                                                                              \n"
    "        int start = 0;                                                                                                                             \n"
    "        int stop = 0;                                                                                                                              \n"
    "                                                                                                                                                   \n"
    "        bool local_bit;                                                                                                                            \n"
    "        vT sum = 0;                                                                                                                                \n"
    "                                                                                                                                                   \n"
    "        int offset_pointer = empty_rows ? d_partition_descriptor_offset_pointer[par_id] : 0;                                                       \n"
    "                                                                                                                                                   \n"
    "        uiT descriptor = d_partition_descriptor[lane_id];                                                                                          \n"
    "                                                                                                                                                   \n"
    "        int y_offset = descriptor >> (32 - bit_y_offset);                                                                                          \n"
    "        const int scansum_offset = (descriptor << bit_y_offset) >> (32 - bit_scansum_offset);                                                      \n"
    "        const int bit_bitflag = 32 - bit_y_offset - bit_scansum_offset;                                                                            \n"
    "                                                                                                                                                   \n"
    "        bool direct = false;                                                                                                                       \n"
    "                                                                                                                                                   \n"
    "        vT first_sum, last_sum;                                                                                                                    \n"
    "                                                                                                                                                   \n"
    "        // step 1. thread-level seg sum                                                                                                            \n"
    "    #if ANONYMOUSLIB_CSR5_SIGMA > 16                                                                                                                   \n"
    "        int ly = 0;                                                                                                                                \n"
    "    #endif                                                                                                                                         \n"
    "                                                                                                                                                   \n"
    "        // extract the first bit-flag packet                                                                                                       \n"
    "        descriptor = descriptor << (bit_y_offset + bit_scansum_offset);                                                                            \n"
    "        descriptor = lane_id ? descriptor : descriptor | 0x80000000;                                                                               \n"
    "                                                                                                                                                   \n"
    "        local_bit = (descriptor >> 31) & 0x1;                                                                                                      \n"
    "        start = !local_bit;                                                                                                                        \n"
    "        direct = local_bit & (bool)lane_id;                                                                                                        \n"
    "                                                                                                                                                   \n"
    "        sum = candidate(d_value_partition, d_x, d_column_index_partition, lane_id, alpha);                                                         \n"
    "                                                                                                                                                   \n"
    "        #pragma unroll                                                                                                                             \n"
    "        for (int i = 1; i < ANONYMOUSLIB_CSR5_SIGMA; i++)                                                                                              \n"
    "        {                                                                                                                                          \n"
    "    #if ANONYMOUSLIB_CSR5_SIGMA > 16                                                                                                                   \n"
    "            int norm_i = i - bit_bitflag;                                                                                                          \n"
    "                                                                                                                                                   \n"
    "            if (!(ly || norm_i) || (ly && !(31 & norm_i)))                                                                                         \n"
    "            {                                                                                                                                      \n"
    "                ly++;                                                                                                                              \n"
    "                descriptor = d_partition_descriptor[ly * ANONYMOUSLIB_CSR5_OMEGA + lane_id];                                                           \n"
    "            }                                                                                                                                      \n"
    "            norm_i = !ly ? 31 & i : 31 & norm_i;                                                                                                   \n"
    "            norm_i = 31 - norm_i;                                                                                                                  \n"
    "                                                                                                                                                   \n"
    "            local_bit = (descriptor >> norm_i) & 0x1;                                                                                              \n"
    "    #else                                                                                                                                          \n"
    "            local_bit = (descriptor >> 31-i) & 0x1;                                                                                                \n"
    "    #endif                                                                                                                                         \n"
    "            if (local_bit)                                                                                                                         \n"
    "            {                                                                                                                                      \n"
    "                if (direct)                                                                                                                        \n"
    "                    d_y[empty_rows ? d_partition_descriptor_offset[offset_pointer + y_offset] : y_offset] = sum;                                   \n"
    "                else                                                                                                                               \n"
    "                    first_sum = sum;                                                                                                               \n"
    "            }                                                                                                                                      \n"
    "                                                                                                                                                   \n"
    "            y_offset += local_bit & direct;                                                                                                        \n"
    "                                                                                                                                                   \n"
    "            direct |= local_bit;                                                                                                                   \n"
    "            sum = local_bit ? 0 : sum;                                                                                                             \n"
    "            stop += local_bit;                                                                                                                     \n"
    "                                                                                                                                                   \n"
    "            sum += candidate(d_value_partition, d_x, d_column_index_partition, i * ANONYMOUSLIB_CSR5_OMEGA + lane_id, alpha);                          \n"
    "        }                                                                                                                                          \n"
    "                                                                                                                                                   \n"
    "        first_sum = direct ? first_sum : sum;                                                                                                      \n"
    "        last_sum = sum;                                                                                                                            \n"
    "                                                                                                                                                   \n"
    "        // step 2. segmented sum                                                                                                                   \n"
    "        sum = start ? first_sum : 0;                                                                                                               \n"
    "                                                                                                                                                   \n"
    "        sum = segmented_sum(sum, s_sum, scansum_offset, lane_id);                                                                                  \n"
    "                                                                                                                                                   \n"
    "        // step 3-1. add s_sum to position stop                                                                                                    \n"
    "        last_sum += (start <= stop) ? sum : 0;                                                                                                     \n"
    "                                                                                                                                                   \n"
    "        // step 3-2. write sums to result array                                                                                                    \n"
    "        if (direct)                                                                                                                                \n"
    "            d_y[empty_rows ? d_partition_descriptor_offset[offset_pointer + y_offset] : y_offset] = last_sum;                                      \n"
    "                                                                                                                                                   \n"
    "        // the first/last value of the first thread goes to calibration                                                                            \n"
    "        if (!lane_id)                                                                                                                              \n"
    "            d_calibrator[par_id] = direct ? first_sum : last_sum;                                                                                  \n"
    "    }                                                                                                                                              \n"
    "                                                                                                                                                   \n"
    "    inline                                                                                                                                         \n"
    "    void spmv_partition(__global const iT           *d_column_index_partition,                                                                     \n"
    "                        __global const vT           *d_value_partition,                                                                            \n"
    "                        __global const iT           *d_row_pointer,                                                                                \n"
    "                        __global const vT           *d_x,                                                                                          \n"
    "                        __global const uiT          *d_partition_pointer,                                                                          \n"
    "                        __global const uiT          *d_partition_descriptor,                                                                       \n"
    "                        __global const iT           *d_partition_descriptor_offset_pointer,                                                        \n"
    "                        __global const iT           *d_partition_descriptor_offset,                                                                \n"
    "                        __global vT                 *d_calibrator,                                                                                 \n"
    "                        __global vT                 *d_y,                                                                                          \n"
    "                        const iT            par_id,                                                                                                \n"
    "                        const int           lane_id,                                                                                               \n"
    "                        const int           bunch_id,                                                                                              \n"
    "                        const int           bit_y_offset,                                                                                          \n"
    "                        const int           bit_scansum_offset,                                                                                    \n"
    "                        const vT            alpha,                                                                                                 \n"
    "                        volatile __local vT  *s_sum,                     \n"
    "                        volatile __local int *s_scan,                     \n"
    "                        volatile __local uiT *s_row_start_stop)                     \n"
    "    {                                                                                                                                              \n"
    "        //volatile __local vT s_y[ANONYMOUSLIB_THREAD_GROUP];                                                                                          \n"
    "                                                                                                                                                   \n"
    "        //volatile __local vT  s_sum[ANONYMOUSLIB_THREAD_GROUP + ANONYMOUSLIB_CSR5_OMEGA / 2];                                                               \n"
    "        //volatile __local int s_scan[(ANONYMOUSLIB_CSR5_OMEGA + 1) * (ANONYMOUSLIB_THREAD_GROUP / ANONYMOUSLIB_CSR5_OMEGA)];                                    \n"
    "                                                                                                                                                   \n"
    "        uiT row_start, row_stop;                                                                                                                   \n"
    "                                                                                                                                                   \n"
    "        //volatile __local uiT s_row_start_stop[ANONYMOUSLIB_THREAD_GROUP / ANONYMOUSLIB_CSR5_OMEGA + 1];                                                    \n"
    "        if (get_local_id(0) < ANONYMOUSLIB_THREAD_GROUP / ANONYMOUSLIB_CSR5_OMEGA + 1)                                                                     \n"
    "            s_row_start_stop[get_local_id(0)] = d_partition_pointer[par_id + get_local_id(0)];                                                     \n"
    "        barrier(CLK_LOCAL_MEM_FENCE);                                                                                                              \n"
    "                                                                                                                                                   \n"
    "        row_start = s_row_start_stop[bunch_id];                                                                                                    \n"
    "        row_stop  = s_row_start_stop[bunch_id + 1] & 0x7FFFFFFF;                                                                                   \n"
    "                                                                                                                                                   \n"
    "        if (row_start == row_stop) // fast track through reduction                                                                                 \n"
    "        {                                                                                                                                          \n"
    "            partition_fast_track                                                                                                                   \n"
    "                    (d_value_partition, d_x, d_column_index_partition,                                                                             \n"
    "                     d_calibrator,                                                                                                                 \n"
    "                     &s_sum[bunch_id * ANONYMOUSLIB_CSR5_OMEGA],                                                                                       \n"
    "                     lane_id, par_id, alpha);                                                                                                      \n"
    "        }                                                                                                                                          \n"
    "        else                                                                                                                                       \n"
    "        {                                                                                                                                          \n"
    "            const bool empty_rows = (row_start >> 31) & 0x1;                                                                                       \n"
    "            row_start &= 0x7FFFFFFF;                                                                                                               \n"
    "                                                                                                                                                   \n"
    "            d_y = &d_y[row_start+1];                                                                                                               \n"
    "                                                                                                                                                   \n"
    "            partition_normal_track                                                                                                                 \n"
    "                    (d_column_index_partition, d_value_partition, d_x,                                                                             \n"
    "                     d_partition_descriptor, d_partition_descriptor_offset_pointer, d_partition_descriptor_offset,                                 \n"
    "                     d_calibrator, d_y,                                                                                                            \n"
    "                     &s_sum[bunch_id * ANONYMOUSLIB_CSR5_OMEGA],                                                                                       \n"
    "                     &s_scan[bunch_id * (ANONYMOUSLIB_CSR5_OMEGA + 1)],                                                                                \n"
    "                     par_id, lane_id,                                                                                                              \n"
    "                     bit_y_offset, bit_scansum_offset, row_start, empty_rows, alpha);                                                              \n"
    "        }                                                                                                                                          \n"
    "    }                                                                                                                                              \n"
    "                                                                                                                                                   \n"
    "    __kernel                                                                                                                                       \n"
    "    void spmv_csr5_compute_kernel(__global const iT           *d_column_index,                                                                     \n"
    "                                  __global const vT           *d_value,                                                                            \n"
    "                                  __global const iT           *d_row_pointer,                                                                      \n"
    "                                  __global const vT           *d_x,                                                                                \n"
    "                                  __global const uiT          *d_partition_pointer,                                                                \n"
    "                                  __global const uiT          *d_partition_descriptor,                                                             \n"
    "                                  __global const iT           *d_partition_descriptor_offset_pointer,                                              \n"
    "                                  __global const iT           *d_partition_descriptor_offset,                                                      \n"
    "                                  __global vT                 *d_calibrator,                                                                       \n"
    "                                  __global vT                 *d_y,                                                                                \n"
    "                                  const iT            p,                                                                                           \n"
    "                                  const int           num_packet,                                                                                  \n"
    "                                  const int           bit_y_offset,                                                                                \n"
    "                                  const int           bit_scansum_offset,                                                                          \n"
    "                                  const vT            alpha)                                                                                       \n"
    "    {                                                                                                                                              \n"
    "        // warp lane id                                                                                                                            \n"
    "        const int lane_id = get_local_id(0) % ANONYMOUSLIB_CSR5_OMEGA;                                                         \n"
    "        // warp global id == par_id                                                                                                                \n"
    "        const iT  par_id = get_global_id(0) / ANONYMOUSLIB_CSR5_OMEGA;                                                                                 \n"
    "        const int bunch_id = get_local_id(0) / ANONYMOUSLIB_CSR5_OMEGA;                                                                                \n"
    "                                                                                                                                                   \n"
    "        if (par_id >= p - 1)                                                                                                                       \n"
    "            return;                                                                                                                                \n"
    "        volatile __local vT  s_sum[ANONYMOUSLIB_THREAD_GROUP + ANONYMOUSLIB_CSR5_OMEGA / 2];                   \n"
    "        volatile __local int s_scan[(ANONYMOUSLIB_CSR5_OMEGA + 1) * (ANONYMOUSLIB_THREAD_GROUP / ANONYMOUSLIB_CSR5_OMEGA)];                   \n"
    "        volatile __local uiT s_row_start_stop[ANONYMOUSLIB_THREAD_GROUP / ANONYMOUSLIB_CSR5_OMEGA + 1];                   \n"
    "                                                                                                                                                   \n"
    "        spmv_partition(&d_column_index[par_id * ANONYMOUSLIB_CSR5_OMEGA * ANONYMOUSLIB_CSR5_SIGMA],                                                          \n"
    "                     &d_value[par_id * ANONYMOUSLIB_CSR5_OMEGA * ANONYMOUSLIB_CSR5_SIGMA],                                                                 \n"
    "                     d_row_pointer, d_x, d_partition_pointer,                                                                                      \n"
    "                     &d_partition_descriptor[par_id * ANONYMOUSLIB_CSR5_OMEGA * num_packet],                                                           \n"
    "                     d_partition_descriptor_offset_pointer, d_partition_descriptor_offset,                                                         \n"
    "                     d_calibrator, d_y,                                                                                                            \n"
    "                     par_id, lane_id, bunch_id, bit_y_offset, bit_scansum_offset, alpha, s_sum, s_scan, s_row_start_stop);                         \n"
    "    }                                                                                                                                              \n"
    "                                                                                                                                                   \n"
    "    __kernel                                                                                                                                       \n"
    "    void spmv_csr5_calibrate_kernel(__global const uiT *d_partition_pointer,                                                                       \n"
    "                                    __global const vT  *d_calibrator,                                                                              \n"
    "                                    __global vT        *d_y,                                                                                       \n"
    "                                    const iT   p)                                                                                                  \n"
    "    {                                                                                                                                              \n"
    "        const int lane_id  = get_local_id(0) % ANONYMOUSLIB_THREAD_BUNCH;                                                                              \n"
    "        const int bunch_id = get_local_id(0) / ANONYMOUSLIB_THREAD_BUNCH;                                                                              \n"
    "        const int local_id = get_local_id(0);                                                                                                      \n"
    "        const iT global_id = get_global_id(0);                                                                                                     \n"
    "                                                                                                                                                   \n"
    "        vT sum;                                                                                                                                    \n"
    "                                                                                                                                                   \n"
    "        volatile __local iT s_partition_pointer[ANONYMOUSLIB_THREAD_GROUP+1];                                                                          \n"
    "        volatile __local vT  s_calibrator[ANONYMOUSLIB_THREAD_GROUP];                                                                                  \n"
    "        //volatile __local vT  s_sum[ANONYMOUSLIB_THREAD_GROUP / ANONYMOUSLIB_THREAD_BUNCH];                                                               \n"
    "                                                                                                                                                   \n"
    "        s_partition_pointer[local_id] = global_id < p-1 ? d_partition_pointer[global_id] & 0x7FFFFFFF : -1;                                        \n"
    "        s_calibrator[local_id] = sum = global_id < p-1 ? d_calibrator[global_id] : 0;                                                              \n"
    "        barrier(CLK_LOCAL_MEM_FENCE);                                                                                                              \n"
    "                                                                                                                                                   \n"
    "        // do a fast track if all s_partition_pointer are the same                                                                                 \n"
    "        if (s_partition_pointer[0] == s_partition_pointer[ANONYMOUSLIB_THREAD_GROUP-1])                                                                \n"
    "        {                                                                                                                                          \n"
    "            // sum all calibrators                                                                                                                 \n"
    "            sum_256(s_calibrator, local_id);                                                                                                       \n"
    "            //d_y[s_partition_pointer[0]] += sum;                                                                                                  \n"
    "            if (!local_id)                                                                                                                         \n"
    "            {                                                                                                                                      \n"
    "                if (sizeof(vT) == 8)                                                                                                               \n"
    "                  atom_add_fp64(&d_y[s_partition_pointer[0]], s_calibrator[0]);                                                                    \n"
    "                else                                                                                                                               \n"
    "                  atom_add_fp32(&d_y[s_partition_pointer[0]], s_calibrator[0]);                                                                    \n"
    "            }                                                                                                                                      \n"
    "            return;                                                                                                                                \n"
    "        }                                                                                                                                          \n"
    "                                                                                                                                                   \n"
    "        int local_par_id = local_id;                                                                                                               \n"
    "        iT row_start_current, row_start_target, row_start_previous;                                                                                \n"
    "        sum = 0;                                                                                                                                   \n"
    "                                                                                                                                                   \n"
    "        // use (p - 1), due to the tail partition is dealt with CSR-vector method                                                                  \n"
    "        if (global_id < p - 1)                                                                                                                     \n"
    "        {                                                                                                                                          \n"
    "            row_start_previous = local_id ? s_partition_pointer[local_id-1] : -1;                                                                  \n"
    "            row_start_current = s_partition_pointer[local_id];                                                                                     \n"
    "                                                                                                                                                   \n"
    "            if (row_start_previous != row_start_current)                                                                                           \n"
    "            {                                                                                                                                      \n"
    "                row_start_target = row_start_current;                                                                                              \n"
    "                                                                                                                                                   \n"
    "                while (row_start_target == row_start_current && local_par_id < get_local_size(0))                                                  \n"
    "                {                                                                                                                                  \n"
    "                    sum +=  s_calibrator[local_par_id];                                                                                            \n"
    "                    local_par_id++;                                                                                                                \n"
    "                    row_start_current = s_partition_pointer[local_par_id];                                                                         \n"
    "                }                                                                                                                                  \n"
    "                                                                                                                                                   \n"
    "                if (row_start_target == s_partition_pointer[0] || row_start_target == s_partition_pointer[ANONYMOUSLIB_THREAD_GROUP-1])                \n"
    "                {                                                                                                                                  \n"
    "                    if (sizeof(vT) == 8)                                                                                                           \n"
    "                        atom_add_fp64(&d_y[row_start_target], sum);                                                                                \n"
    "                    else                                                                                                                           \n"
    "                        atom_add_fp32(&d_y[row_start_target], sum);                                                                                \n"
    "                }                                                                                                                                  \n"
    "                else                                                                                                                               \n"
    "                    d_y[row_start_target] += sum;                                                                                                  \n"
    "            }                                                                                                                                      \n"
    "        }                                                                                                                                          \n"
    "    }                                                                                                                                              \n"
    "                                                                                                                                                   \n"
    "    __kernel                                                                                                                                       \n"
    "    void spmv_csr5_tail_partition_kernel(__global const iT           *d_row_pointer,                                                               \n"
    "                                         __global const iT           *d_column_index,                                                              \n"
    "                                         __global const vT           *d_value,                                                                     \n"
    "                                         __global const vT           *d_x,                                                                         \n"
    "                                         __global vT                 *d_y,                                                                         \n"
    "                                         const iT            tail_partition_start,                                                                 \n"
    "                                         const iT            p,                                                                                    \n"
    "                                         const int           sigma,                                                                                \n"
    "                                         const vT            alpha)                                                                                \n"
    "    {                                                                                                                                              \n"
    "        const int local_id = get_local_id(0);                                                                                                      \n"
    "                                                                                                                                                   \n"
    "        const iT row_id    = tail_partition_start + get_group_id(0);                                                                               \n"
    "        const iT row_start = !get_group_id(0) ? (p - 1) * ANONYMOUSLIB_CSR5_OMEGA * sigma : d_row_pointer[row_id];                                     \n"
    "        const iT row_stop  = d_row_pointer[row_id + 1];                                                                                            \n"
    "                                                                                                                                                   \n"
    "        vT sum = 0;                                                                                                                                \n"
    "                                                                                                                                                   \n"
    "        for (iT idx = local_id + row_start; idx < row_stop; idx += ANONYMOUSLIB_CSR5_OMEGA)                                                            \n"
    "            sum += candidate(d_value, d_x, d_column_index, idx, alpha);                                                                            \n"
    "                                                                                                                                                   \n"
    "        volatile __local vT s_sum[ANONYMOUSLIB_CSR5_OMEGA + ANONYMOUSLIB_CSR5_OMEGA / 2];                                                                  \n"
    "        s_sum[local_id] = sum;                                                                                                                     \n"
    "        sum_64(s_sum, local_id);                                                                                                                   \n"
    "        sum = s_sum[local_id];                                                                                                                     \n"
    "                                                                                                                                                   \n"
    "        if (!local_id)                                                                                                                             \n"
    "            d_y[row_id] = !get_group_id(0) ? d_y[row_id] + sum : sum;                                                                              \n"
    "    }                                                                                                                                              \n"
    "    \n";

//    BasicCL basicCL;
//    cl_program cpSpMV;
//    err  = basicCL.getProgramFromFile(&cpSpMV, _ocl_context, "format_kernels.cl");
//    if(err != CL_SUCCESS)
//        cout << "BUILD ERROR = " << err << endl;
//    else
//        cout << "BUILD SUCCESS" << endl;

    return err;
}

#endif // ANONYMOUSLIB_OPENCL_H
