/*
 *  cuda_utils.h
 *
 *  Created by Dwight Bell on 5/14/10.
 *  Copyright 2010 dbelll. All rights reserved.
 *
 */

#ifndef __CUDA_UTILS_H__
#define __CUDA_UTILS_H__

#include <stdio.h>
#include <cuda.h>
#include "cutil.h"
#include "cuda_runtime.h"
#include "cutil_inline.h"

// define this symbol to print a message at all device_copyx calls
//#define TRACE_DEVICE_ALLOCATIONS

// macros to read command line arguments or use a default value
// This macro assums argc and argv are their normal values found in main()
// It also uses temporary static values __iTemp an integer and __fTemp a float

// put the symbol PREPARE_PARAM at the top of the file that will use the macros
#define PREPARE_PARAM static int __iTemp = 0; static float __fTemp = 0.0f;

// use these macros inside a function where argc and argv are in scope
#define GET_PARAM(str, default) (CUTTrue == cutGetCmdLineArgumenti(argc, argv, (str), &__iTemp)) ? __iTemp : (default)
#define GET_PARAMF(str, default) (CUTTrue == cutGetCmdLineArgumentf(argc, argv, (str), &__fTemp)) ? __fTemp : (default)
#define PARAM_PRESENT(str) (CUTTrue == cutCheckCmdLineFlag(argc, argv, (str)))

// allocate room on the device and copy data from host, returning the device pointer
// returned pointer must be ultimately freed on the device
float *device_copyf(float *data, unsigned count);
unsigned *device_copyui(unsigned *data, unsigned count);

// allocate room on the device, returning the device pointer
// returned pointer must be ultimately freed on the device
float *device_allocf(unsigned count);
unsigned *device_allocui(unsigned count);

// allocate room on the host and copy data from device, returning the host pointer
// returned pointer must be ultimately freed on the host
float *host_copyf(float *d_data, unsigned count);
unsigned *host_copyui(unsigned *d_data, unsigned count);

// Macros for calculating timing values.
// Caller must supply a pointer to unsigned int when creating a timer,
// and the unsigned int for other timer calls.
void CREATE_TIMER(unsigned int *p_timer);
void START_TIMER(unsigned int timer);
float STOP_TIMER(unsigned int timer, char *message);
void DELETE_TIMER(unsigned int timer);

#endif
