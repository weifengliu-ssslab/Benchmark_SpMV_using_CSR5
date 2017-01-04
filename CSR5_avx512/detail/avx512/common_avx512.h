#ifndef COMMON_AVX512_H
#define COMMON_AVX512_H

#include <stdint.h>

#include <omp.h>
#include "immintrin.h"
#include "zmmintrin.h"

#include "../common.h"
#include "../utils.h"

#define ANONYMOUSLIB_CSR5_OMEGA   8
#define ANONYMOUSLIB_CSR5_SIGMA   12
#define ANONYMOUSLIB_X86_CACHELINE   64

#endif // COMMON_AVX512_H
