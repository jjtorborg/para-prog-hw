// CUDA Methods Header

#ifndef GOL_WITH_CUDA
#define GOL_WITH_CUDA

#include <stdio.h>
#include <stdlib.h>

extern void gol_kernelLaunch(size_t iterationsCount, ushort threadsCount);

extern inline void gol_initMaster(unsigned int pattern, size_t worldWidth, size_t worldHeight, int myrank);

extern inline void gol_printWorld();

#endif
