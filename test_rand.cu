//
//  test_rand.cu
//  test_rand
//
//  Created by Dwight Bell on 9/15/10.
//  Copyright dbelll 2010. All rights reserved.
//

#include <cuda.h>
#include "cutil.h"

#include "test_rand.h"
#include "cuda_utils.h"

#include "cuda_rand.cu"


static unsigned num = 0;
static unsigned normalFlag = 0;

__constant__ unsigned dc_normalFlag;

PREPARE_PARAM

#pragma mark process command line
void get_params(int argc, const char **argv)
{
	num = GET_PARAM("NUM", 100);
	normalFlag = PARAM_PRESENT("NORMAL");
}

void dump_params()
{
	printf("parameters:\n");
	printf("--NUM=%d\n", num);
	if (normalFlag) printf("--NORMAL\n");
}

unsigned getNumRands() { return num; }
unsigned getNormalFlag() { return normalFlag; }


#pragma mark CUDA Kernels

/*
*	generate n sets of seed values, storing them at d_seeds
*	use g_seeds as the initial seed to generate the other seeds
*/
__global__ void seeds_kernel(unsigned n, unsigned *d_seeds, unsigned *g_seeds)
{
	__shared__ unsigned s_seeds[BLOCK_SIZE * 4];
	
	// put g_seeds into shared memory
	unsigned idx = threadIdx.x;
	if(idx == 0){
		s_seeds[0] = g_seeds[0];
		s_seeds[BLOCK_SIZE] = g_seeds[1];
		s_seeds[2*BLOCK_SIZE] = g_seeds[2];
		s_seeds[3*BLOCK_SIZE] = g_seeds[3];		
	}
	__syncthreads();
	
	// generate seeds in shared memory for each thread in the block
	unsigned seed_count = 1;
	while (seed_count < BLOCK_SIZE) {
		if ((idx < seed_count) && ((idx + seed_count) < BLOCK_SIZE)){
			s_seeds[idx + seed_count] = RandUniformui(s_seeds + idx, BLOCK_SIZE);
			s_seeds[idx + seed_count + BLOCK_SIZE] = RandUniformui(s_seeds + idx, BLOCK_SIZE);
			s_seeds[idx + seed_count + 2*BLOCK_SIZE] = RandUniformui(s_seeds + idx, BLOCK_SIZE);
			s_seeds[idx + seed_count + 3*BLOCK_SIZE] = RandUniformui(s_seeds + idx, BLOCK_SIZE);
		}
		seed_count *= 2;
		__syncthreads();
	}
	
	// should now have BLOCK_SIZE seeds generated
	// Now use all threads in the block and generate all the seeds
	for(unsigned i = 0; i < (1 + (n-1)/BLOCK_SIZE); i++){
		if ((i*BLOCK_SIZE + idx) < n){
			d_seeds[i * BLOCK_SIZE + idx] = RandUniformui(s_seeds+idx, BLOCK_SIZE);
			d_seeds[i * BLOCK_SIZE + idx + n] = RandUniformui(s_seeds+idx, BLOCK_SIZE);
			d_seeds[i * BLOCK_SIZE + idx + 2*n] = RandUniformui(s_seeds+idx, BLOCK_SIZE);
			d_seeds[i * BLOCK_SIZE + idx + 3*n] = RandUniformui(s_seeds+idx, BLOCK_SIZE);
		}
	}
}


/*
*	generate n random floats between 0 and 1 using the 4*n seeds at d_seeds
*	put the results at d_rand on the device
*/
__global__ void rand_kernel(unsigned n, unsigned *d_seeds, float *d_rand)
{
	unsigned iGlobal = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * gridDim.x * blockDim.x;

	if (iGlobal < n) {
		if (dc_normalFlag) {
			d_rand[iGlobal] = RandNorm(d_seeds + iGlobal, n);
		}else {
			d_rand[iGlobal] = RandUniform(d_seeds + iGlobal, n);
		}
	}
}


#pragma mark CPU front end
__host__ float *generateCPU(unsigned n, unsigned *h_seeds)
{
	unsigned timer;
	CREATE_TIMER(&timer);
	
	// allocate room for random numbers
	float *rands = (float *)malloc(n * sizeof(float));
	
	START_TIMER(timer);
	if (normalFlag) {
		for (unsigned i=0; i < n; i++) {
			rands[i] = RandNorm(h_seeds + i, n);
		}
	}else {
		for (unsigned i=0; i < n; i++) {
			rands[i] = RandUniform(h_seeds + i, n);
		}
	}

	STOP_TIMER(timer, "generate random numbers on CPU");
	
	return rands;
}

#pragma mark GPU front end
/*
	Initialize memory on GPU and return the pointer to the seeds
*/
unsigned * initGPU(unsigned n, unsigned *g_seeds)
{
	unsigned timer;
	CREATE_TIMER(&timer);
	
	unsigned *d_seeds = device_allocui(4 * n);
	unsigned *d_g_seeds = device_copyui(g_seeds, 4);
	
	dim3 blockDim(BLOCK_SIZE);
	dim3 gridDim(1);
	
	START_TIMER(timer);
	seeds_kernel<<<gridDim, blockDim>>>(n, d_seeds, d_g_seeds);
	cudaThreadSynchronize();
	STOP_TIMER(timer, "generate seeds on GPU");
	
	cudaFree(d_g_seeds);
	return d_seeds;
}

unsigned *copySeedsToHost(unsigned n, unsigned *d_seeds)
{
	unsigned timer;
	CREATE_TIMER(&timer);
	
	START_TIMER(timer);
	unsigned *result = host_copyui(d_seeds, 4*n);
	cudaThreadSynchronize();
	STOP_TIMER(timer, "copy seeds to host");
	return result;
}

/*
	generate random nubmers using the given array of seeds
*/
float *generateGPU(unsigned n, unsigned *d_seeds)
{
	unsigned timer;
	CREATE_TIMER(&timer);
	
	cudaMemcpyToSymbol("dc_normalFlag", &normalFlag, sizeof(unsigned));
	
	dim3 blockDim(BLOCK_SIZE);
	dim3 gridDim(1 + (n-1) / BLOCK_SIZE);
	if (gridDim.x > 65535) {
		gridDim.y = (1 + (gridDim.x - 1) / 65535);
		gridDim.x = (1 + (gridDim.x - 1) / gridDim.y);
	}
//	printf("%d threads per block, grid is (%d by %d)\n", blockDim.x, gridDim.x, gridDim.y);
	
	// allocate room for the random numbers on the device
	float *d_rands = device_allocf(n);
	
	START_TIMER(timer);
	rand_kernel<<<gridDim, blockDim>>>(n, d_seeds, d_rands);
	cudaThreadSynchronize();
	STOP_TIMER(timer, "generate random numbers on GPU");
	
	float *h_rands = host_copyf(d_rands, n);
	
	cudaFree(d_rands);
	return h_rands;
}

void free_seeds(unsigned *d_seeds)
{
	cudaFree(d_seeds);
}

void dump_seeds(unsigned n, unsigned *seeds, const char *message)
{
	printf("%s\n", message);
	for (int i = 0; i < n; i++) {
		printf("[seeds%10u]%12u%12u%12u%12u\n", i, seeds[i], seeds[i+n], seeds[i+2*n], seeds[i+3*n]);
	}
}

void dump_host_seeds(unsigned n, unsigned *seeds)
{
	dump_seeds(n, seeds, "host seeds:");
}

void dump_device_seeds(unsigned n, unsigned *d_seeds)
{
	unsigned *h_seeds = host_copyui(d_seeds, 4*n);
	dump_seeds(n, h_seeds, "device seeds:");
	free(h_seeds);
}
