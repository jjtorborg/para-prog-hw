// Para Prog: Assignment 3 GOL MPI

// C Standard headers
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>

// MPI Header
#include <mpi.h>

// Entry point
int main(int argc, char *argv[])
{
    unsigned int pattern = 0;
    unsigned int worldSize = 0;
    unsigned int iterations = 0;
    unsigned int threads = 0;
    unsigned int outputOn = 0;

    //printf("This is the Game of Life running in parallel on GPU(s).\n");

    if (argc != 6)
    {
        fprintf(stderr, "GOL requires 5 arguments: pattern number, sq size of the world, the number of iterations, the number of threads per block and if output is on e.g. ./gol 0 32 2 2 1\n");
        exit(-1);
    }

    pattern = atoi(argv[1]);
    worldSize = atoi(argv[2]);
    iterations = atoi(argv[3]);
    threads = atoi(argv[4]);
    outputOn = atoi(argv[5]);

    int myrank = 0;
    int numranks = 0;

    // Initialize the MPI environment
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &numranks);

    // Start time with MPI_Wtime
    double startTime = MPI_Wtime();

    // Allocate myrank chunk per pattern (dont forget ghost rows)

    // Init the world
    gol_initMaster(pattern, worldSize, worldSize, myrank);

    // Run simluatoin for input number of iterations
    for (int i = 0; i < iterations; i++)
    {
        // Exchange row data with MPI ranks
        unsigned char *firstGhostRow;
        unsigned char *secondGhostRow;
        
        // MPI_Isend/Irecv

        // Launches the parallel computation of the world
        gol_kernelLaunch(threads);

        // Need to call device synchronize (maybe, at some point)
        cudaDeviceSynchronize();
    }

    // Need to call this
    MPI_Barrier(MPI_COMM_WORLD);

    // End MPI_Wtime measurement
    if (myrank == 0)
    {
        double endTime = MPI_Wtime();

        // Output the total time
        printf("Elapsed time: %d\n", endTime - startTime);
    }
                
    // Print statements for case of output on
    if (outputOn) {
        printf("######################### FINAL WORLD IS ###############################\n");                                                                  
        gol_printWorld();
    }

    // Need to call this
    MPI_Finalize();

    // Cleanup any shared memory after finished
    cudaCleanup();

    return EXIT_SUCCESS;
}
