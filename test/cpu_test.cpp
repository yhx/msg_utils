
#include <stdio.h>
#include <stdlib.h>

#include "../msg_utils/CrossMap.h"
#include "../msg_utils/CrossSpike.h"

const int DELAY = 3;
const int N = 4;
const uinteger_t CAP = 8;

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

	CrossMap map(N, N, proc_num);

	for (int i=0; i<N; i++) {
		map._idx2index[i] = i;
	}

	switch (proc_rank) {
		case 0:
			for (int s=0; s<N; s++) {
				if (s != proc_rank) {
					map._index2ridx[s*proc_num+s] = N + proc_rank;
				}
			}
			break;
		case 1:
			for (int s=0; s<N; s++) {
				if (s != proc_rank) {
					map._index2ridx[s*proc_num+s] = N + proc_rank;
				}
			}
			break;
		case 2:
			for (int s=0; s<N; s++) {
				if (s != proc_rank) {
					map._index2ridx[s*proc_num+s] = N + proc_rank;
				}
			}
			break;
		case 3:
			for (int s=0; s<N; s++) {
				if (s != proc_rank) {
					map._index2ridx[s*proc_num+s] = N + proc_rank;
				}
			}
			break;
		default:
			printf("Test case should carry out on four processes\n");
			exit(-1);
			break;
	}

	
	uinteger_t table[(DELAY+1) * CAP] = {
		0, 0, 0, 0,
		1, 2, 0, 0,
		2, 3, 0, 0,
		1, 2, 3, 4
	};

	uinteger_t table_sizes[DELAY+1] = {1, 2, 3, 4};

	CrossSpike cs(proc_rank, proc_num, DELAY);
	cs._recv_offset[0] = 0;
	cs._send_offset[0] = 0;

	for (int i=0; i<proc_num; i++) {
		cs._recv_offset[i+1] = cs._recv_offset[i] + DELAY;
		cs._send_offset[i+1] = cs._send_offset[i] + DELAY;
	}

	cs.alloc();

	for (int t=0; t<DELAY; t++) {
		cs.fetch_cpu(&map, (uinteger_t *)table, (uinteger_t *)table_sizes, CAP, proc_num, DELAY, t);
		cs.update_cpu(t);
		cs.upload_cpu((uinteger_t *)table, (uinteger_t *)table_sizes, CAP, DELAY, t);
	}

	MPI_Finalize();
}
