//
//  test_rand.h
//  test_rand
//
//  Created by Dwight Bell on 9/15/10.
//  Copyright dbelll 2010. All rights reserved.
//

#define BLOCK_SIZE 512


unsigned *initGPU(unsigned n, unsigned *g_seeds);
unsigned *copySeedsToHost(unsigned n, unsigned *d_seeds);
float *generateCPU(unsigned n, unsigned *h_seeds);
float *generateGPU(unsigned n, unsigned *d_seeds);
void free_seeds(unsigned *d_seeds);
void analyze_results(unsigned n, float *randsCPU, float *randsGPU);
void dump_host_seeds(unsigned n, unsigned *seeds);
void dump_device_seeds(unsigned n, unsigned *d_seeds);

void get_params(int argc, const char **argv);
void dump_params();

unsigned getNumRands();
unsigned getNormalFlag();
