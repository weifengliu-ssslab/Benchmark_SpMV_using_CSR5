#ifndef FORMAT_H
#define FORMAT_H

#include "common_opencl.h"
#include "utils_opencl.h"

int format_warmup(cl_kernel           ocl_kernel_warmup,
                  cl_context          ocl_context,
                  cl_command_queue    ocl_command_queue)
{
    int err = ANONYMOUSLIB_SUCCESS;

    cl_mem d_scan;
    d_scan = clCreateBuffer(ocl_context, CL_MEM_READ_WRITE, ANONYMOUSLIB_CSR5_OMEGA * sizeof(int), NULL, &err);

    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    int num_threads = ANONYMOUSLIB_CSR5_OMEGA;
    int num_blocks  = 4000;

    szLocalWorkSize[0]  = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    err  = clSetKernelArg(ocl_kernel_warmup, 0,  sizeof(cl_mem), (void*)&d_scan);

    for (int i = 0; i < 50; i++)
    {

        err = clEnqueueNDRangeKernel(ocl_command_queue, ocl_kernel_warmup, 1,
                                     NULL, szGlobalWorkSize, szLocalWorkSize, 0, NULL, NULL);
        if(err != CL_SUCCESS) { cout << "ocl_kernel_warmup kernel run error = " << err << endl; return err; }
    }

    if(d_scan) err = clReleaseMemObject(d_scan); if(err != CL_SUCCESS) return err;

    return err;
}

template<typename ANONYMOUSLIB_IT, typename ANONYMOUSLIB_UIT>
int generate_partition_pointer(cl_kernel           ocl_kernel_generate_partition_pointer_s1,
                               cl_kernel           ocl_kernel_generate_partition_pointer_s2,
                               cl_command_queue    ocl_command_queue,
                               const int           sigma,
                               const ANONYMOUSLIB_IT   p,
                               const ANONYMOUSLIB_IT   m,
                               const ANONYMOUSLIB_IT   nnz,
                               cl_mem partition_pointer,
                               cl_mem row_pointer,
                               double                    *time)
{
    int err = ANONYMOUSLIB_SUCCESS;

    double conv_time = 0;

    BasicCL basicCL;
    cl_event            ceTimer;                 // OpenCL event
    cl_ulong            queuedTime;
    cl_ulong            submitTime;
    cl_ulong            startTime;
    cl_ulong            endTime;

    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    int num_threads = 128;
    int num_blocks  = ceil((double)(p + 1) / (double)num_threads);

    szLocalWorkSize[0]  = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    // step 1. binary search row pointer
    err  = clSetKernelArg(ocl_kernel_generate_partition_pointer_s1, 0, sizeof(cl_mem), (void*)&row_pointer);
    err |= clSetKernelArg(ocl_kernel_generate_partition_pointer_s1, 1, sizeof(cl_mem), (void*)&partition_pointer);
    err |= clSetKernelArg(ocl_kernel_generate_partition_pointer_s1, 2, sizeof(cl_int), (void*)&sigma);
    err |= clSetKernelArg(ocl_kernel_generate_partition_pointer_s1, 3, sizeof(cl_int), (void*)&p);
    err |= clSetKernelArg(ocl_kernel_generate_partition_pointer_s1, 4, sizeof(cl_int), (void*)&m);
    err |= clSetKernelArg(ocl_kernel_generate_partition_pointer_s1, 5, sizeof(cl_int), (void*)&nnz);

    err = clEnqueueNDRangeKernel(ocl_command_queue, ocl_kernel_generate_partition_pointer_s1, 1,
                                 NULL, szGlobalWorkSize, szLocalWorkSize, 0, NULL, &ceTimer);
    if(err != CL_SUCCESS) { cout << "ocl_kernel_generate_partition_pointer_s1 kernel run error = " << err << endl; return err; }

    err = clWaitForEvents(1, &ceTimer);
    if(err != CL_SUCCESS) { cout << "event error = " << err << endl; return err; }

    basicCL.getEventTimer(ceTimer, &queuedTime, &submitTime, &startTime, &endTime);
    conv_time += double(endTime - startTime) / 1000000.0;

    // step 2. check empty rows
    num_threads = 64;
    num_blocks  = p;

    szLocalWorkSize[0]  = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    err  = clSetKernelArg(ocl_kernel_generate_partition_pointer_s2, 0, sizeof(cl_mem), (void*)&row_pointer);
    err |= clSetKernelArg(ocl_kernel_generate_partition_pointer_s2, 1, sizeof(cl_mem), (void*)&partition_pointer);

    err = clEnqueueNDRangeKernel(ocl_command_queue, ocl_kernel_generate_partition_pointer_s2, 1,
                                 NULL, szGlobalWorkSize, szLocalWorkSize, 0, NULL, &ceTimer);
    if(err != CL_SUCCESS) { cout << "ocl_kernel_generate_partition_pointer_s2 kernel run error = " << err << endl; return err; }

    err = clWaitForEvents(1, &ceTimer);
    if(err != CL_SUCCESS) { cout << "event error = " << err << endl; return err; }

    basicCL.getEventTimer(ceTimer, &queuedTime, &submitTime, &startTime, &endTime);
    conv_time += double(endTime - startTime) / 1000000.0;

//    ANONYMOUSLIB_UIT *debug = (ANONYMOUSLIB_UIT *)malloc((p + 1) * sizeof(ANONYMOUSLIB_UIT));
//    err = clEnqueueReadBuffer(ocl_command_queue, row_pointer, CL_TRUE,
//                              0, (p + 1) * sizeof(ANONYMOUSLIB_UIT), debug, 0, NULL, NULL);
//    if(err != CL_SUCCESS) return err;

//    for (int i = 0; i < 64; i++)
//        cout << "debug[ " << i << "] = " << debug[i] << endl;

    *time = conv_time;

    return err;
}

template<typename ANONYMOUSLIB_IT, typename ANONYMOUSLIB_UIT>
int generate_partition_descriptor(cl_kernel           ocl_kernel_generate_partition_descriptor_s0,
                                  cl_kernel           ocl_kernel_generate_partition_descriptor_s1,
                                  cl_kernel           ocl_kernel_generate_partition_descriptor_s2,
                                  cl_kernel           ocl_kernel_generate_partition_descriptor_s3,
                                  cl_command_queue    ocl_command_queue,
                                  const int           sigma,
                                  const ANONYMOUSLIB_IT   p,
                                  const ANONYMOUSLIB_IT   m,
                                  const int           bit_y_offset,
                                  const int           bit_scansum_offset,
                                  const int           num_packet,
                                  cl_mem              row_pointer,
                                  cl_mem              partition_pointer,
                                  cl_mem              partition_descriptor,
                                  cl_mem              partition_descriptor_offset_pointer,
                                  int                *_num_offsets,
                                  double                    *time)
{
    int err = ANONYMOUSLIB_SUCCESS;

    double conv_time = 0;

    BasicCL basicCL;
    cl_event            ceTimer;                 // OpenCL event
    cl_ulong            queuedTime;
    cl_ulong            submitTime;
    cl_ulong            startTime;
    cl_ulong            endTime;

    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    int num_threads = ANONYMOUSLIB_CSR5_OMEGA;
    int num_blocks = p;

    szLocalWorkSize[0]  = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    err  = clSetKernelArg(ocl_kernel_generate_partition_descriptor_s0, 0, sizeof(cl_mem), (void*)&partition_descriptor);
    err |= clSetKernelArg(ocl_kernel_generate_partition_descriptor_s0, 1, sizeof(cl_int), (void*)&num_packet);

    err = clEnqueueNDRangeKernel(ocl_command_queue, ocl_kernel_generate_partition_descriptor_s0, 1,
                                 NULL, szGlobalWorkSize, szLocalWorkSize, 0, NULL, &ceTimer);
    if(err != CL_SUCCESS) { cout << "ocl_kernel_generate_partition_descriptor_s0 kernel run error = " << err << endl; return err; }

    err = clWaitForEvents(1, &ceTimer);
    if(err != CL_SUCCESS) { cout << "event error = " << err << endl; return err; }

    basicCL.getEventTimer(ceTimer, &queuedTime, &submitTime, &startTime, &endTime);
    conv_time += double(endTime - startTime) / 1000000.0;



    num_threads = 128;
    num_blocks = ceil((double)m / (double)num_threads);

    szLocalWorkSize[0]  = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    int bit_all_offset = bit_y_offset + bit_scansum_offset;

    err  = clSetKernelArg(ocl_kernel_generate_partition_descriptor_s1, 0, sizeof(cl_mem), (void*)&row_pointer);
    err |= clSetKernelArg(ocl_kernel_generate_partition_descriptor_s1, 1, sizeof(cl_mem), (void*)&partition_descriptor);
    err |= clSetKernelArg(ocl_kernel_generate_partition_descriptor_s1, 2, sizeof(cl_int), (void*)&m);
    err |= clSetKernelArg(ocl_kernel_generate_partition_descriptor_s1, 3, sizeof(cl_int), (void*)&sigma);
    err |= clSetKernelArg(ocl_kernel_generate_partition_descriptor_s1, 4, sizeof(cl_int), (void*)&bit_all_offset);
    err |= clSetKernelArg(ocl_kernel_generate_partition_descriptor_s1, 5, sizeof(cl_int), (void*)&num_packet);

    err = clEnqueueNDRangeKernel(ocl_command_queue, ocl_kernel_generate_partition_descriptor_s1, 1,
                                 NULL, szGlobalWorkSize, szLocalWorkSize, 0, NULL, &ceTimer);
    if(err != CL_SUCCESS) { cout << "ocl_kernel_generate_partition_descriptor_s1 kernel run error = " << err << endl; return err; }

    err = clWaitForEvents(1, &ceTimer);
    if(err != CL_SUCCESS) { cout << "event error = " << err << endl; return err; }

    basicCL.getEventTimer(ceTimer, &queuedTime, &submitTime, &startTime, &endTime);
    conv_time += double(endTime - startTime) / 1000000.0;




    num_threads = ANONYMOUSLIB_THREAD_GROUP;
    num_blocks  = ceil ((double)(p-1) / (double)(num_threads / ANONYMOUSLIB_CSR5_OMEGA));

    szLocalWorkSize[0]  = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    err  = clSetKernelArg(ocl_kernel_generate_partition_descriptor_s2, 0, sizeof(cl_mem), (void*)&partition_pointer);
    err |= clSetKernelArg(ocl_kernel_generate_partition_descriptor_s2, 1, sizeof(cl_mem), (void*)&partition_descriptor);
    err |= clSetKernelArg(ocl_kernel_generate_partition_descriptor_s2, 2, sizeof(cl_mem), (void*)&partition_descriptor_offset_pointer);
    err |= clSetKernelArg(ocl_kernel_generate_partition_descriptor_s2, 3, sizeof(cl_int), (void*)&sigma);
    err |= clSetKernelArg(ocl_kernel_generate_partition_descriptor_s2, 4, sizeof(cl_int), (void*)&num_packet);
    err |= clSetKernelArg(ocl_kernel_generate_partition_descriptor_s2, 5, sizeof(cl_int), (void*)&bit_y_offset);
    err |= clSetKernelArg(ocl_kernel_generate_partition_descriptor_s2, 6, sizeof(cl_int), (void*)&bit_scansum_offset);
    err |= clSetKernelArg(ocl_kernel_generate_partition_descriptor_s2, 7, sizeof(cl_int), (void*)&p);

    err = clEnqueueNDRangeKernel(ocl_command_queue, ocl_kernel_generate_partition_descriptor_s2, 1,
                                 NULL, szGlobalWorkSize, szLocalWorkSize, 0, NULL, &ceTimer);
    if(err != CL_SUCCESS) { cout << "ocl_kernel_generate_partition_descriptor_s2 kernel run error = " << err << endl; return err; }

    err = clWaitForEvents(1, &ceTimer);
    if(err != CL_SUCCESS) { cout << "event error = " << err << endl; return err; }

    basicCL.getEventTimer(ceTimer, &queuedTime, &submitTime, &startTime, &endTime);
    conv_time += double(endTime - startTime) / 1000000.0;

    ANONYMOUSLIB_IT num_offsets = 0;

//    err = clEnqueueReadBuffer(ocl_command_queue, partition_descriptor_offset_pointer, CL_TRUE,
//                              p * sizeof(ANONYMOUSLIB_IT), sizeof(ANONYMOUSLIB_IT), &num_offsets, 0, NULL, &ceTimer);
//    if(err != CL_SUCCESS) return err;
//    err = clWaitForEvents(1, &ceTimer);
//    if(err != CL_SUCCESS) { cout << "event error = " << err << endl; return err; }

//    basicCL.getEventTimer(ceTimer, &queuedTime, &submitTime, &startTime, &endTime);
//    conv_time += double(endTime - startTime) / 1000000.0;

//    cout << "num_offsets = " << num_offsets << endl;

    //clFinish(ocl_command_queue);

//    if (num_offsets)
//    {
        // prefix-sum partition_descriptor_offset_pointer
        num_threads = 256;
        num_blocks  = 1;

        szLocalWorkSize[0]  = num_threads;
        szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

        err  = clSetKernelArg(ocl_kernel_generate_partition_descriptor_s3, 0, sizeof(cl_mem), (void*)&partition_descriptor_offset_pointer);
        err |= clSetKernelArg(ocl_kernel_generate_partition_descriptor_s3, 1, sizeof(cl_int), (void*)&p);

        err = clEnqueueNDRangeKernel(ocl_command_queue, ocl_kernel_generate_partition_descriptor_s3, 1,
                                     NULL, szGlobalWorkSize, szLocalWorkSize, 0, NULL, &ceTimer);
        if(err != CL_SUCCESS) { cout << "ocl_kernel_generate_partition_descriptor_s3 kernel run error = " << err << endl; return err; }

        err = clWaitForEvents(1, &ceTimer);
        if(err != CL_SUCCESS) { cout << "event error = " << err << endl; return err; }

        basicCL.getEventTimer(ceTimer, &queuedTime, &submitTime, &startTime, &endTime);
        conv_time += double(endTime - startTime) / 1000000.0;

        err = clEnqueueReadBuffer(ocl_command_queue, partition_descriptor_offset_pointer, CL_TRUE,
                                  p * sizeof(ANONYMOUSLIB_IT), sizeof(ANONYMOUSLIB_IT), &num_offsets, 0, NULL, &ceTimer);
        if(err != CL_SUCCESS) return err;
        err = clWaitForEvents(1, &ceTimer);
        if(err != CL_SUCCESS) { cout << "event error = " << err << endl; return err; }

        basicCL.getEventTimer(ceTimer, &queuedTime, &submitTime, &startTime, &endTime);
        conv_time += double(endTime - startTime) / 1000000.0;

        //clFinish(ocl_command_queue);
//    }

    *_num_offsets = num_offsets;
    *time = conv_time;

    return err;
}

template<typename ANONYMOUSLIB_IT, typename ANONYMOUSLIB_UIT>
int generate_partition_descriptor_offset(cl_kernel           ocl_kernel_generate_partition_descriptor_offset,
                                         cl_command_queue    ocl_command_queue,
                                         const int           sigma,
                                         const ANONYMOUSLIB_IT   p,
                                         const int           bit_y_offset,
                                         const int           bit_scansum_offset,
                                         const int           num_packet,
                                         cl_mem              row_pointer,
                                         cl_mem              partition_pointer,
                                         cl_mem              partition_descriptor,
                                         cl_mem              partition_descriptor_offset_pointer,
                                         cl_mem              partition_descriptor_offset,
                                         double                    *time)
{
    int err = ANONYMOUSLIB_SUCCESS;

    double conv_time = 0;

    BasicCL basicCL;
    cl_event            ceTimer;                 // OpenCL event
    cl_ulong            queuedTime;
    cl_ulong            submitTime;
    cl_ulong            startTime;
    cl_ulong            endTime;

    if (p <= 1) return err;

    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    int num_threads = ANONYMOUSLIB_THREAD_GROUP;
    int num_blocks = ceil ((double)(p-1) / (double)(num_threads / ANONYMOUSLIB_CSR5_OMEGA));

    szLocalWorkSize[0]  = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    err  = clSetKernelArg(ocl_kernel_generate_partition_descriptor_offset, 0, sizeof(cl_mem), (void*)&row_pointer);
    err |= clSetKernelArg(ocl_kernel_generate_partition_descriptor_offset, 1, sizeof(cl_mem), (void*)&partition_pointer);
    err |= clSetKernelArg(ocl_kernel_generate_partition_descriptor_offset, 2, sizeof(cl_mem), (void*)&partition_descriptor);
    err |= clSetKernelArg(ocl_kernel_generate_partition_descriptor_offset, 3, sizeof(cl_mem), (void*)&partition_descriptor_offset_pointer);
    err |= clSetKernelArg(ocl_kernel_generate_partition_descriptor_offset, 4, sizeof(cl_mem), (void*)&partition_descriptor_offset);
    err |= clSetKernelArg(ocl_kernel_generate_partition_descriptor_offset, 5, sizeof(cl_int), (void*)&p);
    err |= clSetKernelArg(ocl_kernel_generate_partition_descriptor_offset, 6, sizeof(cl_int), (void*)&num_packet);
    err |= clSetKernelArg(ocl_kernel_generate_partition_descriptor_offset, 7, sizeof(cl_int), (void*)&bit_y_offset);
    err |= clSetKernelArg(ocl_kernel_generate_partition_descriptor_offset, 8, sizeof(cl_int), (void*)&bit_scansum_offset);
    err |= clSetKernelArg(ocl_kernel_generate_partition_descriptor_offset, 9, sizeof(cl_int), (void*)&sigma);

    err = clEnqueueNDRangeKernel(ocl_command_queue, ocl_kernel_generate_partition_descriptor_offset, 1,
                                 NULL, szGlobalWorkSize, szLocalWorkSize, 0, NULL, &ceTimer);
    if(err != CL_SUCCESS) { cout << "ocl_kernel_generate_partition_descriptor_offset kernel run error = " << err << endl; return err; }

    err = clWaitForEvents(1, &ceTimer);
    if(err != CL_SUCCESS) { cout << "event error = " << err << endl; return err; }

    basicCL.getEventTimer(ceTimer, &queuedTime, &submitTime, &startTime, &endTime);
    conv_time += double(endTime - startTime) / 1000000.0;

    *time = conv_time;

    return err;
}

template<typename ANONYMOUSLIB_IT, typename ANONYMOUSLIB_UIT, typename ANONYMOUSLIB_VT>
int aosoa_transpose(cl_kernel           _ocl_kernel_aosoa_transpose_smem_iT,
                    cl_kernel           _ocl_kernel_aosoa_transpose_smem_vT,
                    cl_command_queue    ocl_command_queue,
                    const int           sigma,
                    const int           nnz,
                    cl_mem              partition_pointer,
                    cl_mem              column_index,
                    cl_mem              value,
                    int                R2C,
                    double                    *time)
{
    int err = ANONYMOUSLIB_SUCCESS;

    double conv_time = 0;

    BasicCL basicCL;
    cl_event            ceTimer;                 // OpenCL event
    cl_ulong            queuedTime;
    cl_ulong            submitTime;
    cl_ulong            startTime;
    cl_ulong            endTime;

    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    int num_threads = 128;
    int num_blocks = ceil((double)nnz / (double)(ANONYMOUSLIB_CSR5_OMEGA * sigma)) - 1;

    szLocalWorkSize[0]  = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    err  = clSetKernelArg(_ocl_kernel_aosoa_transpose_smem_iT, 0, sizeof(cl_mem), (void*)&column_index);
    err |= clSetKernelArg(_ocl_kernel_aosoa_transpose_smem_iT, 1, sizeof(cl_mem), (void*)&partition_pointer);
    err |= clSetKernelArg(_ocl_kernel_aosoa_transpose_smem_iT, 2, sizeof(cl_int), (void*)&R2C);

    err = clEnqueueNDRangeKernel(ocl_command_queue, _ocl_kernel_aosoa_transpose_smem_iT, 1,
                                 NULL, szGlobalWorkSize, szLocalWorkSize, 0, NULL, &ceTimer);
    if(err != CL_SUCCESS) { cout << "_ocl_kernel_aosoa_transpose_smem_iT kernel run error = " << err << endl; return err; }

    err = clWaitForEvents(1, &ceTimer);
    if(err != CL_SUCCESS) { cout << "event error = " << err << endl; return err; }

    basicCL.getEventTimer(ceTimer, &queuedTime, &submitTime, &startTime, &endTime);
    conv_time += double(endTime - startTime) / 1000000.0;



    err  = clSetKernelArg(_ocl_kernel_aosoa_transpose_smem_vT, 0, sizeof(cl_mem), (void*)&value);
    err |= clSetKernelArg(_ocl_kernel_aosoa_transpose_smem_vT, 1, sizeof(cl_mem), (void*)&partition_pointer);
    err |= clSetKernelArg(_ocl_kernel_aosoa_transpose_smem_vT, 2, sizeof(cl_int), (void*)&R2C);

    err = clEnqueueNDRangeKernel(ocl_command_queue, _ocl_kernel_aosoa_transpose_smem_vT, 1,
                                 NULL, szGlobalWorkSize, szLocalWorkSize, 0, NULL, &ceTimer);
    if(err != CL_SUCCESS) { cout << "_ocl_kernel_aosoa_transpose_smem_vT kernel run error = " << err << endl; return err; }

    err = clWaitForEvents(1, &ceTimer);
    if(err != CL_SUCCESS) { cout << "event error = " << err << endl; return err; }

    basicCL.getEventTimer(ceTimer, &queuedTime, &submitTime, &startTime, &endTime);
    conv_time += double(endTime - startTime) / 1000000.0;

    *time = conv_time;

    return err;
}

#endif // FORMAT_H
