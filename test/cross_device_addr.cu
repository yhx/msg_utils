
#include <stdio.h>

#include "mpi.h"

#include "../helper/helper_gpu.h"

#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);                      \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

int main(int argc, char **argv)
{
	const int NGPU = 2;
	int rank = 0, size = 0;

	MPICHECK(MPI_Init(&argc, &argv));
	MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
	MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));

	gpuSetDevice(rank);

	int *cpu = new int[100];
	int *res = new int[100];
	int *gpu = NULL;


	for (int i=0; i<100; i++) {
		if (1 == rank) {
			cpu[i] = 2*i+1;
		}
	}

	if (1 == rank) {
		gpu = TOGPU(cpu, 100);
		printf("P1 GPU Pointer: %p\n", gpu);
	}

	MPI_Barrier();

	MPI_Status status;

	if (0 == rank) {
		MPICHECK(MPI_Recv(&gpu, sizeof(int*), MPI_UNSIGNED_CHAR, 1, 0, MPI_COMM_WORLD, &status));
	} else {
		MPICHECK(MPI_Send(&gpu, sizeof(int*), MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD));
	}

	MPI_Barrier();
	if (0 == rank) {
		printf("P0 GPU Pointer: %p\n", gpu);
	}

	MPI_Barrier();
	if (0 == rank) {
		COPYFROMGPU(res, gpu, 100);

		printf("P0 res: ");
		for (int i=0; i<100; i++) {
			printf("%d ", res[i]);
		}
		printf("\n");
	}

	MPICHECK(MPI_Finalize());

	return 0;
}
