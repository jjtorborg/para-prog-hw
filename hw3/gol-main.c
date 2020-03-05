// Para Prog: Assignment 3 GOL MPI

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

    // 

    gol_initMaster(pattern, worldSize, worldSize, myrank);

    // Launches the parallel computation of the world for a defined number of iterations
    gol_kernelLaunch(iterations, threads);
                
    // Print statements for case of output on
    if (outputOn) {
        printf("######################### FINAL WORLD IS ###############################\n");                                                                  
        gol_printWorld();
    }

    // Cleanup any shared memory after finished
    cudaCleanup();

    return EXIT_SUCCESS;
}
