#ifndef COMMON_CUDA_H
#define COMMON_CUDA_H

#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#include "../common.h"
#include "../utils.h"

#define ANONYMOUSLIB_CSR5_OMEGA   32
#define ANONYMOUSLIB_THREAD_BUNCH 32
#define ANONYMOUSLIB_THREAD_GROUP 128

#define ANONYMOUSLIB_AUTO_TUNED_SIGMA -1

#endif // COMMON_CUDA_H
