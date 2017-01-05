#ifndef COMMON_H
#define COMMON_H

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <math.h>

using namespace std;

#define ANONYMOUSLIB_SUCCESS                   0
#define ANONYMOUSLIB_UNKOWN_FORMAT            -1
#define ANONYMOUSLIB_UNSUPPORTED_CSR5_OMEGA   -2
#define ANONYMOUSLIB_CSR_TO_CSR5_FAILED       -3
#define ANONYMOUSLIB_UNSUPPORTED_CSR_SPMV     -4
#define ANONYMOUSLIB_UNSUPPORTED_VALUE_TYPE   -5

#define ANONYMOUSLIB_FORMAT_CSR  0
#define ANONYMOUSLIB_FORMAT_CSR5 1
#define ANONYMOUSLIB_FORMAT_HYB5 2

#endif // COMMON_H
