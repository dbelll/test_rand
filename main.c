//
//  main.c
//  test_rand
//
//  Created by Dwight Bell on 9/15/10.
//  Copyright dbelll 2010. All rights reserved.
//


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "test_rand.h"

/*
	Test random number functions on GPU and CPU and make sure they produce identical results.
 
	Create large arry of 
 */

static unsigned g_seeds[4] = {2784565659u, 1491908209u, 3415062841u, 3293636241u};

void analyze_results(unsigned n, float *randsCPU, float *randsGPU)
{
	float totCPU = randsCPU[0];
	float totCPU2 = randsCPU[0] * randsCPU[0];
	float totGPU = randsGPU[0];
	float totGPU2 = randsGPU[0] * randsGPU[0];
	float maxDiff = (randsCPU[0] > randsGPU[0]) ? randsCPU[0] - randsGPU[0] 
												: randsGPU[0] - randsCPU[0]; 
	float minCPU = randsCPU[0];
	float maxCPU = randsCPU[0];
	float minGPU = randsGPU[0];
	float maxGPU = randsGPU[0];
	for (int i = 1; i < n; i++) {
		totCPU += randsCPU[i];
		totCPU2 += randsCPU[i] * randsCPU[i];
		
		totGPU += randsGPU[i];
		totGPU2 += randsGPU[i] * randsGPU[i];
		
		if (minCPU > randsCPU[i]) minCPU = randsCPU[i];
		if (maxCPU < randsCPU[i]) maxCPU = randsCPU[i];
		if (minGPU > randsGPU[i]) minGPU = randsGPU[i];
		if (maxGPU < randsGPU[i]) maxGPU = randsGPU[i];
		
		float diff = randsCPU[i] - randsGPU[i];
		if (diff < 0.0f) diff = -diff;
		if (diff > maxDiff) maxDiff = diff;
	}
	float meanCPU = totCPU/n;
	float meanGPU = totGPU/n;
	float sdCPU = sqrtf(totCPU2/n - meanCPU * meanCPU);
	float sdGPU = sqrtf(totGPU2/n - meanGPU * meanGPU);
	printf("CPU avg =%10.6f, GPU avg =%10.6f, max difference =%9.6f\n", meanCPU, meanGPU, maxDiff);
	printf("CPU  sd =%10.6f,  GPU sd = %10.6f\n", sdCPU, sdGPU);
	printf("CPU min =%10.6f, GPU min =%10.6f\n", minCPU, minGPU);
	printf("CPU max =%10.6f, GPU max =%10.6f\n", maxCPU, maxGPU);
}

int main(int argc, const char **argv)
{
	get_params(argc, argv);
//	dump_params();
	unsigned n = getNumRands();
	unsigned normalFlag = getNormalFlag();
	
	printf("generating%12d random numbers on device and CPU, normalFlag=%d\n", n, normalFlag);
	
	// generate the seeds on GPU based on starting seed value,
	// and then copy seeds back to GPU.
//	printf("generate seeds on GPU...");
	unsigned *d_seeds = initGPU(n, g_seeds);
//	printf(" done\n");
//	printf("copy seeds to CPU...");
	unsigned *h_seeds = copySeedsToHost(n, d_seeds);
//	printf(" done\n");
	
//	dump_host_seeds(n, h_seeds);
//	dump_device_seeds(n, d_seeds);
	
	// generate random numbers on CPU
	float *randsCPU = generateCPU(n, h_seeds);
	
	// generate random numbers on GPU and copy back to CPU, free all memory on GPU.
	float *randsGPU = generateGPU(n, d_seeds);
	
	// compare the results
	analyze_results(n, randsCPU, randsGPU);
	
	free(h_seeds);
	free(randsCPU);
	free(randsGPU);
	
	free_seeds(d_seeds);
	
	return 0;
}
