
#define CATCH_CONFIG_RUNNER

#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "catch.hpp"

#include "../msg_utils/CrossMap.h"
#include "../msg_utils/CrossSpike.h"

using std::vector;

const int DELAY = 3;
const int N = 4;
const uinteger_t CAP = 8;

int proc_rank = -1;

integer_t table[(DELAY+1) * CAP] = {
	0, 0, 0, 0,
	1, 2, 0, 0,
	2, 3, 0, 0,
	1, 2, 3, 0
};

integer_t table_sizes[DELAY+1] = {1, 2, 3, 4};

TEST_CASE("CHECK RESULTS", "") {
	switch (proc_rank) {
		case 0:
			CHECK_THAT(vector<integer_t>(table[0], table[0] + table_sizes[0]), 
					Catch::UnorderedEquals(vector<integer_t>{0, 4, 5, 6}));
			CHECK_THAT(vector<integer_t>(table[1], table[1] + table_sizes[1]), 
					Catch::UnorderedEquals(vector<integer_t>{1, 2}));
			CHECK_THAT(vector<integer_t>(table[2], table[2] + table_sizes[2]), 
					Catch::UnorderedEquals(vector<integer_t>{0, 2, 3, 4, 5, 6}));
			CHECK_THAT(vector<integer_t>(table[3], table[3] + table_sizes[3]), 
					Catch::UnorderedEquals(vector<integer_t>{0, 1, 2, 3, 4, 5, 6}));
			break;
		case 1:
			CHECK_THAT(vector<integer_t>(table[0], table[0] + table_sizes[0]), 
					Catch::UnorderedEquals(vector<integer_t>{0}));
			CHECK_THAT(vector<integer_t>(table[1], table[1] + table_sizes[1]), 
					Catch::UnorderedEquals(vector<integer_t>{1, 2, 4, 5, 6}));
			CHECK_THAT(vector<integer_t>(table[2], table[2] + table_sizes[2]), 
					Catch::UnorderedEquals(vector<integer_t>{0, 2, 3}));
			CHECK_THAT(vector<integer_t>(table[3], table[3] + table_sizes[3]), 
					Catch::UnorderedEquals(vector<integer_t>{0, 1, 2, 3, 4, 5, 6}));
			break;
		case 2:
			CHECK_THAT(vector<integer_t>(table[0], table[0] + table_sizes[0]), 
					Catch::UnorderedEquals(vector<integer_t>{0}));
			CHECK_THAT(vector<integer_t>(table[1], table[1] + table_sizes[1]), 
					Catch::UnorderedEquals(vector<integer_t>{1, 2, 4, 5, 6}));
			CHECK_THAT(vector<integer_t>(table[2], table[2] + table_sizes[2]), 
					Catch::UnorderedEquals(vector<integer_t>{0, 2, 3, 4, 5, 6}));
			CHECK_THAT(vector<integer_t>(table[3], table[3] + table_sizes[3]), 
					Catch::UnorderedEquals(vector<integer_t>{0, 1, 2, 3, 4, 5, 6}));
			break;
		case 3:
			CHECK_THAT(vector<integer_t>(table[0], table[0] + table_sizes[0]), 
					Catch::UnorderedEquals(vector<integer_t>{0}));
			CHECK_THAT(vector<integer_t>(table[1], table[1] + table_sizes[1]), 
					Catch::UnorderedEquals(vector<integer_t>{1, 2}));
			CHECK_THAT(vector<integer_t>(table[2], table[2] + table_sizes[2]), 
					Catch::UnorderedEquals(vector<integer_t>{0, 2, 3, 4, 5, 6}));
			CHECK_THAT(vector<integer_t>(table[3], table[3] + table_sizes[3]), 
					Catch::UnorderedEquals(vector<integer_t>{0, 1, 2, 3, 4, 5, 6}));
			break;
		default:
			printf("Test case should carry out on four processes\n");
			exit(-1);
			break;
	}
}

int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);

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
				if (s == proc_rank) {
					continue;
				} else if (s < proc_rank) {
					map._index2ridx[s*proc_num+s] = N + proc_rank;
				} else {
					map._index2ridx[s*proc_num+s] = N + proc_rank - 1;
				}
			}
			break;
		case 1:
			for (int s=0; s<N; s++) {
				if (s == proc_rank) {
					continue;
				} else if (s < proc_rank) {
					map._index2ridx[s*proc_num+s] = N + proc_rank;
				} else {
					map._index2ridx[s*proc_num+s] = N + proc_rank - 1;
				}
			}
			break;
		case 2:
			for (int s=0; s<N; s++) {
				if (s == proc_rank) {
					continue;
				} else if (s < proc_rank) {
					map._index2ridx[s*proc_num+s] = N + proc_rank;
				} else {
					map._index2ridx[s*proc_num+s] = N + proc_rank - 1;
				}
			}
			break;
		case 3:
			for (int s=0; s<N; s++) {
				if (s == proc_rank) {
					continue;
				} else if (s < proc_rank) {
					map._index2ridx[s*proc_num+s] = N + proc_rank;
				} else {
					map._index2ridx[s*proc_num+s] = N + proc_rank - 1;
				}
			}
			break;
		default:
			printf("Test case should carry out on four processes\n");
			exit(-1);
			break;
	}

	

	CrossSpike cs(proc_rank, proc_num, DELAY);
	cs._recv_offset[0] = 0;
	cs._send_offset[0] = 0;

	for (int i=0; i<proc_num; i++) {
		cs._recv_offset[i+1] = cs._recv_offset[i] + DELAY;
		cs._send_offset[i+1] = cs._send_offset[i] + DELAY;
	}

	cs.alloc();

	for (int t=0; t<DELAY; t++) {
		cs.fetch_cpu(&map, (integer_t *)table, (integer_t *)table_sizes, CAP, proc_num, DELAY, t);
		cs.update_cpu(t);
		cs.upload_cpu((integer_t *)table, (integer_t *)table_sizes, CAP, DELAY, t);
	}


	int result = Catch::Session().run( argc, argv );


	// for (int i=0; i<DELAY+1; i++) {
	// 	printf("Rank %d:%d :", proc_rank, table_sizes[i]);
	// 	for (int j=0; j<table_sizes[i]; j++) {
	// 		printf("%d ", table[j]);
	// 	}
	// 	printf("\n");
	// }

	MPI_Finalize();

	return result;
}
