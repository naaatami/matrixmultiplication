// Multiplies a matrix in parallel using MPI.
// mpicc -g -Wall -o matrix mpi_mm.c
// mpiexec --n <cores> ./matrix <N> <M> <Q>

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>

void printMatrix(int *C, int numRows, int numCol);

int main(int argc, char *argv[])
{
    int N, M, Q;
    int *matrixA, *matrixB, *matrixC;
    int *localAMatrix, *localCMatrix;
    int numberOfProcessors, rank;
    int matrixAWork;
    MPI_Init(&argc, &argv);
    MPI_Comm mainComm = MPI_COMM_WORLD;
    MPI_Comm_size(mainComm, &numberOfProcessors);
    MPI_Comm_rank(mainComm, &rank);

    if(rank == 0)
    {
        N = atoi(argv[1]); // matrix A rows
        M = atoi(argv[2]); // matrix A cols, matrix B rows
        Q = atoi(argv[3]); // matrix B cols

        matrixA = malloc(N * M * sizeof(int));
        matrixB = malloc(M * Q * sizeof(int));
        matrixC = malloc(N * Q * sizeof(int)); // multiplied array

        // check if N is divible by nproc
        if (N % numberOfProcessors != 0){
            fprintf(stderr, "\n N should be divisible by the number of processors.\n");
            MPI_Abort(mainComm, 1);
        }

        // instantiating matrixA and matrixB
        for (int i = 0; i < N; ++i)
        {
            for (int j = 0; j < M; ++j)
            {
                matrixA[i * M + j] = j % 3;
            }
        }

        for (int i = 0; i < M; ++i)
        {
            for (int j = 0; j < Q; ++j)
            {
                matrixB[i * Q + j] = i % 3;
            }
        }

        // amount of rows of N being sent out
        matrixAWork = N / numberOfProcessors;
    }

    double startTime, elapsedTime;
    MPI_Barrier(mainComm);
    startTime = MPI_Wtime();

    // broadcasting matrixAWork, M, and Q to all so they know what to accept
    MPI_Bcast(&matrixAWork, 1, MPI_INT, 0, mainComm);
    MPI_Bcast(&M, 1, MPI_INT, 0, mainComm);
    MPI_Bcast(&Q, 1, MPI_INT, 0, mainComm);

    // allocate memory for matrixB if it wasn't allocated already - then broadcast it out
    if (rank != 0) {
        matrixB = malloc(M * Q * sizeof(int));
    }
    MPI_Bcast(matrixB, M * Q, MPI_INT, 0, mainComm);
    localAMatrix = malloc(matrixAWork * M * sizeof(int));

    // scattering however many rows of A to everyone
    MPI_Scatter(matrixA, matrixAWork * M, MPI_INT, localAMatrix, matrixAWork * M, MPI_INT, 0, mainComm);

    localCMatrix = malloc(matrixAWork * Q * sizeof(int));
    for(int i = 0; i < matrixAWork; i++)
    {
        for(int j = 0; j < Q; j++)
        {
            int sum = 0;
            for(int k = 0; k < M; k++)
            {
                sum = sum + localAMatrix[i * M + k] * matrixB[k * Q + j];
            }
            localCMatrix[i * Q + j] = sum;
        }
    }

    // gathering matrix C
    MPI_Gather(localCMatrix, matrixAWork * Q, MPI_INT, matrixC, matrixAWork * Q, MPI_INT, 0, mainComm);

    MPI_Barrier(mainComm);
    elapsedTime = MPI_Wtime() - startTime;

    if(rank == 0)
    {
        // serially solving as proof of correctness:
        int *serialMatrixC = malloc(N * Q * sizeof(int));
        for(int i = 0; i < N; i++)
        {
            for(int j = 0; j < Q; j++)
            {
                int sum = 0;
                for(int k = 0; k < M; k++)
                {
                    sum = sum + matrixA[i * M + k] * matrixB[k * Q + j];
                }
                serialMatrixC[i * Q + j] = sum;
            }
        }

        if(N <= 20 && M <= 20 && Q <= 20)
        {
            printf("A[%dx%d]:\n", N, M);
            printMatrix(matrixA, N, M);
            printf("\nB[%dx%d]:\n", M, Q);
            printMatrix(matrixB, M, Q);
            printf("\nC[%dx%d]:\n", N, Q);
            printMatrix(matrixC, N, Q);
            printf("\nMatrix C if solved in serial:\n");
            printMatrix(serialMatrixC, N, Q);
        }

        printf("\nTime to solve in parallel: %f\n", elapsedTime);
        free(matrixA);
        free(matrixC);
        free(serialMatrixC);
    }

    free(localAMatrix);
    free(localCMatrix);
    free(matrixB);

    MPI_Finalize();
    return 0;

}

void printMatrix(int *C, int numRows, int numCol)
{
    int i, j;
    for (i = 0; i < numRows; ++i)
    {
        printf("\t");
        for (j = 0; j < numCol; ++j)
        {
            printf("%d ", C[i * numCol + j]);
        }
        printf("\n");
    }
}
