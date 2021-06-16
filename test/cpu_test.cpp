
#include <stdio.h>


const int DELAY = 4;
const int N = 16;

int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);

	int proc_rank = 0;
	int proc_num = 0;
	MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &proc_num);

	char processor_name[MPI_MAX_PROCESSOR_NAME];
	int name_len;
	MPI_Get_processor_name(processor_name, &name_len);
	printf("Processor %s, rank %d out of %d processors\n", processor_name, proc_rank, proc_num);
	
	uinteger_t table[(DELAY+1) * N] = {};
	uinteger_t table_sizes[DELAY+1] = {};

	for (int t=0; t<DELAY; t++) {
	}

}
