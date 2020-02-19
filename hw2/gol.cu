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
void sharedMemoryInit(unsigned char *g_data, size_t g_dataLength)
{
    cudaMallocManaged(&g_data, (g_dataLength * sizeof(unsigned char)));
    memset(&g_data[0], 0, sizeof(g_data));
}

static inline void gol_initAllZeros(size_t worldWidth, size_t worldHeight)
{
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    // init the shared memory to all zeros
    sharedMemoryInit(g_data, g_dataLength);
    sharedMemoryInit(g_resultData, g_dataLength);
}

static inline void gol_initAllOnes(size_t worldWidth, size_t worldHeight)
{
    int i;

    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    // Init the shared memory of the original grid to all zeros
    sharedMemoryInit(g_data, g_dataLength);

    // set all rows of world to true
    for (i = 0; i < g_dataLength; i++)
    {
        g_data[i] = 1;
    }

    // Init the shared memory of the reuslt grid to all zeros
    sharedMemoryInit(g_resultData, g_dataLength);
}

static inline void gol_initOnesInMiddle(size_t worldWidth, size_t worldHeight)
{
    int i;

    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    // Init the shared memory of the original grid to all zeros
    sharedMemoryInit(g_data, g_dataLength);

    // set first 1 rows of world to true
    for (i = 10 * g_worldWidth; i < 11 * g_worldWidth; i++)
    {
        if ((i >= (10 * g_worldWidth + 10)) && (i < (10 * g_worldWidth + 20)))
        {
            g_data[i] = 1;
        }
    }

    // Init the shared memory of the reuslt grid to all zeros
    sharedMemoryInit(g_resultData, g_dataLength);
}

static inline void gol_initOnesAtCorners(size_t worldWidth, size_t worldHeight)
{
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    // Init the shared memory of the original grid to all zeros
    sharedMemoryInit(g_data, g_dataLength);

    g_data[0] = 1;                                                 // upper left
    g_data[worldWidth - 1] = 1;                                    // upper right
    g_data[(worldHeight * (worldWidth - 1))] = 1;                  // lower left
    g_data[(worldHeight * (worldWidth - 1)) + worldWidth - 1] = 1; // lower right

    // Init the shared memory of the reuslt grid to all zeros
    sharedMemoryInit(g_resultData, g_dataLength);
}

static inline void gol_initSpinnerAtCorner(size_t worldWidth, size_t worldHeight)
{
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    // Init the shared memory of the original grid to all zeros
    sharedMemoryInit(g_data, g_dataLength);

    g_data[0] = 1;              // upper left
    g_data[1] = 1;              // upper left +1
    g_data[worldWidth - 1] = 1; // upper right

    // Init the shared memory of the reuslt grid to all zeros
    sharedMemoryInit(g_resultData, g_dataLength);
}

static inline void gol_initMaster(unsigned int pattern, size_t worldWidth, size_t worldHeight)
{
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
static inline void gol_swap(unsigned char **pA, unsigned char **pB)
{
    unsigned char *temp = *pA;
    *pA = *pB;
    *pB = temp;
}

// Return the number of alive cell neighbors for data[x1+y1]
static inline unsigned int gol_countAliveCells(unsigned char *data,
                                               size_t x0,
                                               size_t x1,
                                               size_t x2,
                                               size_t y0,
                                               size_t y1,
                                               size_t y2)
{

    // You write this function - it should return the number of alive cell for data[x1+y1]
    // There are 8 neighbors - see the assignment description for more details.
    unsigned int aliveCellsCount = 0;

    // Upper left
    if (data[x0 + y0]) {
        aliveCellsCount++;
    }

    // Upper middle
    if (data[x1 + y0]) {
        aliveCellsCount++;        
    }

    // Upper right
    if (data[x2 + y0]) {
        aliveCellsCount++;        
    }

    // Middle left
    if (data[x0 + y1]) {
        aliveCellsCount++;        
    }

    // Middle right
    if (data[x2 + y1]) {
        aliveCellsCount++;        
    }

    // Lower left
    if (data[x0 + y2]) {
        aliveCellsCount++;        
    }

    // Lower middle
    if (data[x1 + y2]) {
        aliveCellsCount++;        
    }

    // Lower right
    if (data[x2 + y2]) {
        aliveCellsCount++;        
    }

    return aliveCellsCount;
}

// Don't modify this function or your submitty autograding may incorrectly grade otherwise correct solutions.
static inline void gol_printWorld()
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

// Main CUDA kernel function - handles parallel threading
__global__ void gol_kernel(const unsigned char* d_data,
                           unsigned char* d_resultData,
                           unsigned int worldWidth,
                           unsigned int worldHeight)
{

    // Iterate over each cell of the grid
    for (index = blockIdx.x * blockDim.x + threadIdx.x;
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

        // Compute if d_resultsData[y1 + x] is 0 or 1
        // Cell is currently alive
        if (d_data[x1 + y1]) {

            // Under-population
            if (aliveCellsCount < 2) {
                d_resultData[x1 + y1] = 0;
            }

            // Survives
            if (aliveCellsCount == 2 || aliveCellsCount == 3) {
                d_resultData[x1 + y1] = 1;
            }

            // Over-population
            if (aliveCellsCount > 3) {
                d_resultData[x1 + y1] = 0;
            }
        }

        // Cell is currently dead
        else {
            
            // Reproduction
            if (aliveCellsCount == 3) {
                d_resultData[x1 + y1] = 1;
            }

            // Stays dead
            else {
                d_resultData[x1 + y1] = 0;
            }
        }
    }

    // Swap resultData and data arrays
    gol_swap(&d_data, &d_resultData);
}

// Launches the parallel computation of the world for a defined number of iterations
void gol_kernelLaunch(unsigned char** d_data,
                      unsigned char** d_resultData,
                      size_t worldWidth,
                      size_t worldHeight,
                      size_t iterationsCount,
                      ushort threadsCount)
{
    // Run the kernel for input num iterations over input num threads
    for (size_t i = 0; i < iterationsCount; i++) {

        // Kernel call
        gol_kernel<<<1, threadsCount>>>(*d_data, *d_resultData, worldWidth, worldHeight);
    }

    // Need to call device synchronize before returning to main
    cudaDeviceSynchronize();
} 

int main(int argc, char *argv[])
{
    unsigned int pattern = 0;
    unsigned int worldSize = 0;
    unsigned int iterations = 0;

    printf("This is the Game of Life running in parallel on GPU(s).\n");

    if (argc != 4)
    {
        printf("GOL requires 3 arguments: pattern number, sq size of the world and the number of iterations, e.g. ./gol 0 32 2 \n");
        exit(-1);
    }

    pattern = atoi(argv[1]);
    worldSize = atoi(argv[2]);
    iterations = atoi(argv[3]);
    threads = atoi(argv[4]);

    gol_initMaster(pattern, worldSize, worldSize);

    // Launches the parallel computation of the world for a defined number of iterations
    gol_kernelLaunch(&g_data, &g_resultData, g_worldWidth, g_worldHeight, iterations, threads);
                                                                                                                                                                                        
    printf("######################### FINAL WORLD IS ###############################\n");                                                                  
    gol_printWorld();

    cudaFree(g_data);
    cudaFree(g_resultData);

    return true;
}
