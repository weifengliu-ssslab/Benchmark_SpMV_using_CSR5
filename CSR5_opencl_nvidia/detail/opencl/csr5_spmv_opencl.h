#ifndef CSR5_SPMV_H
#define CSR5_SPMV_H

#include "common_opencl.h"
#include "utils_opencl.h"

template<typename ANONYMOUSLIB_IT, typename ANONYMOUSLIB_UIT, typename ANONYMOUSLIB_VT>
int csr5_spmv(cl_kernel                 ocl_kernel_spmv_csr5_compute,
              cl_kernel                 ocl_kernel_spmv_csr5_calibrate,
              cl_kernel                 ocl_kernel_spmv_csr5_tail_partition,
              cl_command_queue          ocl_command_queue,
              const int                 sigma,
              const ANONYMOUSLIB_IT         p,
              const ANONYMOUSLIB_IT         m,
              const int                 bit_y_offset,
              const int                 bit_scansum_offset,
              const int                 num_packet,
              const cl_mem                    row_pointer,
              const cl_mem                    column_index,
              const cl_mem                    value,
              const cl_mem                    partition_pointer,
              const cl_mem                    partition_descriptor,
              const cl_mem                    partition_descriptor_offset_pointer,
              const cl_mem                    partition_descriptor_offset,
              cl_mem                    calibrator,
              const ANONYMOUSLIB_IT         tail_partition_start,
              const ANONYMOUSLIB_VT         alpha,
              const cl_mem                    x,
              cl_mem                    y,
              double                    *time)
{
    int err = ANONYMOUSLIB_SUCCESS;

    double spmv_time = 0;

    BasicCL basicCL;
    cl_event            ceTimer;                 // OpenCL event
    cl_ulong            queuedTime;
    cl_ulong            submitTime;
    cl_ulong            startTime;
    cl_ulong            endTime;

    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    int num_threads = ANONYMOUSLIB_THREAD_GROUP;
    int num_blocks = ceil ((double)(p-1) / (double)(num_threads / ANONYMOUSLIB_CSR5_OMEGA));

    szLocalWorkSize[0]  = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    err  = clSetKernelArg(ocl_kernel_spmv_csr5_compute, 0, sizeof(cl_mem), (void*)&column_index);
    err |= clSetKernelArg(ocl_kernel_spmv_csr5_compute, 1, sizeof(cl_mem), (void*)&value);
    err |= clSetKernelArg(ocl_kernel_spmv_csr5_compute, 2, sizeof(cl_mem), (void*)&row_pointer);
    err |= clSetKernelArg(ocl_kernel_spmv_csr5_compute, 3, sizeof(cl_mem), (void*)&x);
    err |= clSetKernelArg(ocl_kernel_spmv_csr5_compute, 4, sizeof(cl_mem), (void*)&partition_pointer);
    err |= clSetKernelArg(ocl_kernel_spmv_csr5_compute, 5, sizeof(cl_mem), (void*)&partition_descriptor);
    err |= clSetKernelArg(ocl_kernel_spmv_csr5_compute, 6, sizeof(cl_mem), (void*)&partition_descriptor_offset_pointer);
    err |= clSetKernelArg(ocl_kernel_spmv_csr5_compute, 7, sizeof(cl_mem), (void*)&partition_descriptor_offset);
    err |= clSetKernelArg(ocl_kernel_spmv_csr5_compute, 8, sizeof(cl_mem), (void*)&calibrator);
    err |= clSetKernelArg(ocl_kernel_spmv_csr5_compute, 9, sizeof(cl_mem), (void*)&y);
    err |= clSetKernelArg(ocl_kernel_spmv_csr5_compute, 10, sizeof(ANONYMOUSLIB_IT), (void*)&p);
    err |= clSetKernelArg(ocl_kernel_spmv_csr5_compute, 11, sizeof(cl_int), (void*)&num_packet);
    err |= clSetKernelArg(ocl_kernel_spmv_csr5_compute, 12, sizeof(cl_int), (void*)&bit_y_offset);
    err |= clSetKernelArg(ocl_kernel_spmv_csr5_compute, 13, sizeof(cl_int), (void*)&bit_scansum_offset);
    err |= clSetKernelArg(ocl_kernel_spmv_csr5_compute, 14, sizeof(ANONYMOUSLIB_VT), (void*)&alpha);

    err = clEnqueueNDRangeKernel(ocl_command_queue, ocl_kernel_spmv_csr5_compute, 1,
                                 NULL, szGlobalWorkSize, szLocalWorkSize, 0, NULL, &ceTimer);
    if(err != CL_SUCCESS) { cout << "ocl_kernel_spmv_csr5_compute kernel run error = " << err << endl; return err; }

    err = clWaitForEvents(1, &ceTimer);
    if(err != CL_SUCCESS) { cout << "event error = " << err << endl; return err; }

    basicCL.getEventTimer(ceTimer, &queuedTime, &submitTime, &startTime, &endTime);
    spmv_time += double(endTime - startTime) / 1000000.0;


    num_threads = ANONYMOUSLIB_THREAD_GROUP;
    num_blocks = ceil((double)(p-1)/(double)num_threads);

    szLocalWorkSize[0]  = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    err  = clSetKernelArg(ocl_kernel_spmv_csr5_calibrate, 0, sizeof(cl_mem), (void*)&partition_pointer);
    err |= clSetKernelArg(ocl_kernel_spmv_csr5_calibrate, 1, sizeof(cl_mem), (void*)&calibrator);
    err |= clSetKernelArg(ocl_kernel_spmv_csr5_calibrate, 2, sizeof(cl_mem), (void*)&y);
    err |= clSetKernelArg(ocl_kernel_spmv_csr5_calibrate, 3, sizeof(cl_int), (void*)&p);

    err = clEnqueueNDRangeKernel(ocl_command_queue, ocl_kernel_spmv_csr5_calibrate, 1,
                                 NULL, szGlobalWorkSize, szLocalWorkSize, 0, NULL, &ceTimer);
    if(err != CL_SUCCESS) { cout << "ocl_kernel_spmv_csr5_calibrate kernel run error = " << err << endl; return err; }

    err = clWaitForEvents(1, &ceTimer);
    if(err != CL_SUCCESS) { cout << "event error = " << err << endl; return err; }

    basicCL.getEventTimer(ceTimer, &queuedTime, &submitTime, &startTime, &endTime);
    spmv_time += double(endTime - startTime) / 1000000.0;

    num_threads = ANONYMOUSLIB_CSR5_OMEGA;
    num_blocks = m - tail_partition_start;

    szLocalWorkSize[0]  = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    err  = clSetKernelArg(ocl_kernel_spmv_csr5_tail_partition, 0, sizeof(cl_mem), (void*)&row_pointer);
    err |= clSetKernelArg(ocl_kernel_spmv_csr5_tail_partition, 1, sizeof(cl_mem), (void*)&column_index);
    err |= clSetKernelArg(ocl_kernel_spmv_csr5_tail_partition, 2, sizeof(cl_mem), (void*)&value);
    err |= clSetKernelArg(ocl_kernel_spmv_csr5_tail_partition, 3, sizeof(cl_mem), (void*)&x);
    err |= clSetKernelArg(ocl_kernel_spmv_csr5_tail_partition, 4, sizeof(cl_mem), (void*)&y);
    err |= clSetKernelArg(ocl_kernel_spmv_csr5_tail_partition, 5, sizeof(cl_int), (void*)&tail_partition_start);
    err |= clSetKernelArg(ocl_kernel_spmv_csr5_tail_partition, 6, sizeof(cl_int), (void*)&p);
    err |= clSetKernelArg(ocl_kernel_spmv_csr5_tail_partition, 7, sizeof(cl_int), (void*)&sigma);
    err |= clSetKernelArg(ocl_kernel_spmv_csr5_tail_partition, 8, sizeof(ANONYMOUSLIB_VT), (void*)&alpha);

    err = clEnqueueNDRangeKernel(ocl_command_queue, ocl_kernel_spmv_csr5_tail_partition, 1,
                                 NULL, szGlobalWorkSize, szLocalWorkSize, 0, NULL, &ceTimer);
    if(err != CL_SUCCESS) { cout << "ocl_kernel_spmv_csr5_tail_partition kernel run error = " << err << endl; return err; }

    err = clWaitForEvents(1, &ceTimer);
    if(err != CL_SUCCESS) { cout << "event error = " << err << endl; return err; }

    basicCL.getEventTimer(ceTimer, &queuedTime, &submitTime, &startTime, &endTime);
    spmv_time += double(endTime - startTime) / 1000000.0;

    *time = spmv_time;

    return err;
}

#endif // CSR5_SPMV_H
