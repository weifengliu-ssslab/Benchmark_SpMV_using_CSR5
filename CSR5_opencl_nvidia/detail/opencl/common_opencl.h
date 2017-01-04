#ifndef COMMON_OPENCL_H
#define COMMON_OPENCL_H

#include <cstring>
#include <string>

#ifdef __APPLE__
#include "OpenCL/opencl.h"
#else
#include "CL/opencl.h"
#endif

#include "basiccl.h"

#include "../common.h"
#include "../utils.h"

#define ANONYMOUSLIB_CSR5_OMEGA   32
#define ANONYMOUSLIB_THREAD_BUNCH 32
#define ANONYMOUSLIB_THREAD_GROUP 256

#define ANONYMOUSLIB_AUTO_TUNED_SIGMA -1

#endif // COMMON_OPENCL_H
