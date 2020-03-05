// Para Prog: Assignment 3 GOL CUDA Method Library

// References: https://devblogs.nvidia.com/even-easier-introduction-cuda/
//      Line 285 - uses the equation presented in this article ^^^
//               -> int numBlocks = (N + blockSize - 1) / blockSize;

// C Standard Headers
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>

// CUDA Includes for parallel programming
#include <cuda.h>
#include <cuda_runtime.h>

// Result from last compute of world.
unsigned char *g_resultData = NULL;

// Current state of world.
unsigned char *g_data = NULL;

// Current width of world.
size_t g_worldWidth = 0;

/// Current height of world.
size_t g_worldHeight = 0;

/// Current data length (product of width and height)
size_t g_dataLength = 0; // g_worldWidth * g_worldHeight

// Method for properly zeroing out all shared memory upon init
extern void sharedMemoryInit(unsigned char **d_data, size_t d_dataLength)
{
    cudaMallocManaged(d_data, (d_dataLength * sizeof(unsigned char)));
    memset(*d_data, 0, sizeof(*d_data));
}

extern inline void gol_initAllZeros(size_t worldWidth, size_t worldHeight)
{
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    // init the shared memory to all zeros
    sharedMemoryInit(&g_data, g_dataLength);
    sharedMemoryInit(&g_resultData, g_dataLength);
}

extern inline void gol_initAllOnes(size_t worldWidth, size_t worldHeight)
{
    int i;

    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    // Init the shared memory of the original grid to all zeros
    sharedMemoryInit(&g_data, g_dataLength);

    // set all rows of world to true
    for (i = 0; i < g_dataLength; i++)
    {
        g_data[i] = 1;
    }

    // Init the shared memory of the reuslt grid to all zeros
    sharedMemoryInit(&g_resultData, g_dataLength);
}

extern inline void gol_initOnesInMiddle(size_t worldWidth, size_t worldHeight)
{
    int i;

    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    // Init the shared memory of the original grid to all zeros
    sharedMemoryInit(&g_data, g_dataLength);

    // set first 1 rows of world to true
    for (i = 10 * g_worldWidth; i < 11 * g_worldWidth; i++)
    {
        if ((i >= (10 * g_worldWidth + 10)) && (i < (10 * g_worldWidth + 20)))
        {
            g_data[i] = 1;
        }
    }

    // Init the shared memory of the reuslt grid to all zeros
    sharedMemoryInit(&g_resultData, g_dataLength);
}

extern inline void gol_initOnesAtCorners(size_t worldWidth, size_t worldHeight)
{
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    // Init the shared memory of the original grid to all zeros
    sharedMemoryInit(&g_data, g_dataLength);

    g_data[0] = 1;                                                 // upper left
    g_data[worldWidth - 1] = 1;                                    // upper right
    g_data[(worldHeight * (worldWidth - 1))] = 1;                  // lower left
    g_data[(worldHeight * (worldWidth - 1)) + worldWidth - 1] = 1; // lower right

    // Init the shared memory of the reuslt grid to all zeros
    sharedMemoryInit(&g_resultData, g_dataLength);
}

extern inline void gol_initSpinnerAtCorner(size_t worldWidth, size_t worldHeight)
{
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    // Init the shared memory of the original grid to all zeros
    sharedMemoryInit(&g_data, g_dataLength);

    g_data[0] = 1;              // upper left
    g_data[1] = 1;              // upper left +1
    g_data[worldWidth - 1] = 1; // upper right

    // Init the shared memory of the reuslt grid to all zeros
    sharedMemoryInit(&g_resultData, g_dataLength);
}

extern inline void gol_initMaster(unsigned int pattern, size_t worldWidth, size_t worldHeight, int myrank)
{
    // Init cudaDeviceCount for use later
    int cudaDeviceCount = 0;
    
    if( (cE = cudaGetDeviceCount( &cudaDeviceCount)) != cudaSuccess )
    {
        printf(" Unable to determine cuda device count, error is %d, count is %d\n",cE, cudaDeviceCount );
        exit(-1);
    }
    
    if( (cE = cudaSetDevice( myrank % cudaDeviceCount )) != cudaSuccess )
    {
        printf(" Unable to have rank %d set to cuda device %d, error is %d \n",myrank, (myrank % cudaDeviceCount), cE);
        exit(-1);
    }

    switch (pattern)
    {
    case 0:
        gol_initAllZeros(worldWidth, worldHeight);
        break;

    case 1:
        gol_initAllOnes(worldWidth, worldHeight);
        break;

    case 2:
        gol_initOnesInMiddle(worldWidth, worldHeight);
        break;

    case 3:
        gol_initOnesAtCorners(worldWidth, worldHeight);
        break;

    case 4:
        gol_initSpinnerAtCorner(worldWidth, worldHeight);
        break;

    default:
        printf("Pattern %u has not been implemented \n", pattern);
        exit(-1);
    }
}

// Swap the pointers of pA and pB.
extern inline void gol_swap(unsigned char **pA, unsigned char **pB)
{
    unsigned char *temp = *pA;
    *pA = *pB;
    *pB = temp;
}

// Return the number of alive cell neighbors for data[x1+y1]
__device__ extern inline unsigned int gol_countAliveCells(unsigned char *data,
                                                          size_t x0,
                                                          size_t x1,
                                                          size_t x2,
                                                          size_t y0,
                                                          size_t y1,
                                                          size_t y2)
{

    // Compute the number of alive cells by summing the states of each surrounding cell
    unsigned int aliveCellsCount = data[x0 + y0] +
                                   data[x1 + y0] +
                                   data[x2 + y0] +
                                   data[x0 + y1] +
                                   data[x2 + y1] +
                                   data[x0 + y2] +
                                   data[x1 + y2] +
                                   data[x2 + y2];

    return aliveCellsCount;
}

// Don't modify this function or your submitty autograding may incorrectly grade otherwise correct solutions.
extern inline void gol_printWorld()
{
    int i, j;

    for (i = 0; i < g_worldHeight; i++)
    {
        printf("Row %2d: ", i);
        for (j = 0; j < g_worldWidth; j++)
        {
            printf("%u ", (unsigned int)g_data[(i * g_worldWidth) + j]);
        }
        printf("\n");
    }

    printf("\n\n");
}

// Cleanup CUDA memory
extern inline void cudaCleanup()
{
    // Return any memory to system
    cudaFree(g_data);
    cudaFree(g_resultData);
}

// Main CUDA kernel function - handles parallel threading
__global__ extern void gol_kernel(unsigned char* d_data,
                           unsigned char* d_resultData,
                           unsigned int worldWidth,
                           unsigned int worldHeight)
{

    // Iterate over each cell of the grid
    for (unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < worldWidth * worldHeight;
         index += blockDim.x * gridDim.x) {

        // Use the thread index to calculate current x and y position
        size_t y = index / worldHeight;
        size_t x = index % worldWidth;

        // Compute offsets and perform analyses on cells
        // Set y0, y1 and y2
        size_t y0 = ((y + worldHeight - 1) % worldHeight) * worldWidth;
        size_t y1 = y * worldWidth;
        size_t y2 = ((y + 1) % worldHeight) * worldWidth;

        // Set x0, x1 and x2
        size_t x1 = x;
        size_t x0 = (x1 + worldWidth - 1) % worldWidth;
        size_t x2 = (x1 + 1) % worldWidth;

        // Call countAliveCells
        unsigned int aliveCellsCount = gol_countAliveCells(d_data, x0, x1, x2, y0, y1, y2);

        // Cell is currently alive
        if (d_data[x1 + y1]) {

            // Under-population (curr cell dies)
            if (aliveCellsCount < 2) {
                d_resultData[x1 + y1] = 0;
            }

            // Optimal population (curr cell survives)
            if (aliveCellsCount == 2 || aliveCellsCount == 3) {
                d_resultData[x1 + y1] = 1;
            }

            // Over-population (curr cell dies)
            if (aliveCellsCount > 3) {
                d_resultData[x1 + y1] = 0;
            }
        }

        // Cell is currently dead
        else {
            
            // Reproduction (curr cell becomes alive)
            if (aliveCellsCount == 3) {
                d_resultData[x1 + y1] = 1;
            }

            // Cell stays dead
            else {
                d_resultData[x1 + y1] = 0;
            }
        }
    }
}

// Launches the parallel computation of the world for a defined number of iterations
extern void gol_kernelLaunch(ushort threadsCount)
{
    // Compute block number based on CUDA docs (devblogs.nvidia.com -> see References "line 3")
    int blocksCount = ((g_worldWidth * g_worldHeight) + threadsCount - 1) / threadsCount;

    // Kernel call
    gol_kernel<<<blocksCount, threadsCount>>>(g_data, g_resultData, g_worldWidth, g_worldHeight);

    // Swap resultData and data arrays
    gol_swap(g_data, g_resultData);
} 
