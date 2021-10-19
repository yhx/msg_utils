
#define CATCH_CONFIG_RUNNER

#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "nccl.h"

#include "catch.hpp"

#include "../helper/helper_gpu.h"
#include "../msg_utils/msg_utils.h"
#include "../msg_utils/CrossMap.h"
#include "../msg_utils/CrossSpike.h"
#include "../msg_utils/CrossSpike.h"
#include "../msg_utils/ProcBuf.h"

using std::vector;

#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);                      \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

// const int GPU_SIZE = 2;
const int DELAY = 3;
const int N = 4;
const uinteger_t CAP = 8;
const int THREAD_NUM = 2;

int proc_rank = -1;

nid_t table[(DELAY+1) * CAP] = {
	0, 0, 0, 0, 0, 0, 0, 0,
	1, 2, 0, 0, 0, 0, 0, 0,
	2, 3, 0, 0, 0, 0, 0, 0,
	1, 2, 3, 4, 0, 0, 0, 0
};

nsize_t table_sizes[DELAY+1] = {1, 2, 3, 4};


TEST_CASE("CHECK Update", "") {
	switch (proc_rank) {
		case 0:
			CHECK_THAT(vector<integer_t>(table + 0*CAP, table + 0*CAP + table_sizes[0]), 
					Catch::UnorderedEquals(vector<integer_t>{0, 4, 5, 6}));
			CHECK_THAT(vector<integer_t>(table + 1*CAP, table + 1*CAP + table_sizes[1]), 
					Catch::UnorderedEquals(vector<integer_t>{1, 2}));
			CHECK_THAT(vector<integer_t>(table + 2*CAP, table + 2*CAP + table_sizes[2]), 
					Catch::UnorderedEquals(vector<integer_t>{0, 2, 3, 4, 5, 6}));
			// CHECK_THAT(vector<integer_t>(table + 3*CAP, table + 3*CAP + table_sizes[3]), 
			// 		Catch::UnorderedEquals(vector<integer_t>{0, 1, 2, 3, 4, 5, 6}));
			break;
		case 1:
			CHECK_THAT(vector<integer_t>(table + 0*CAP, table + 0*CAP + table_sizes[0]), 
					Catch::UnorderedEquals(vector<integer_t>{0}));
			CHECK_THAT(vector<integer_t>(table + 1*CAP, table + 1*CAP + table_sizes[1]), 
					Catch::UnorderedEquals(vector<integer_t>{1, 2, 4, 5, 6}));
			CHECK_THAT(vector<integer_t>(table + 2*CAP, table + 2*CAP + table_sizes[2]), 
					Catch::UnorderedEquals(vector<integer_t>{0, 2, 3}));
			// CHECK_THAT(vector<integer_t>(table + 3*CAP, table + 3*CAP + table_sizes[3]), 
			// 		Catch::UnorderedEquals(vector<integer_t>{0, 1, 2, 3, 4, 5, 6}));
			break;
		case 2:
			CHECK_THAT(vector<integer_t>(table + 0*CAP, table + 0*CAP + table_sizes[0]), 
					Catch::UnorderedEquals(vector<integer_t>{0}));
			CHECK_THAT(vector<integer_t>(table + 1*CAP, table + 1*CAP + table_sizes[1]), 
					Catch::UnorderedEquals(vector<integer_t>{1, 2, 4, 5, 6}));
			CHECK_THAT(vector<integer_t>(table + 2*CAP, table + 2*CAP + table_sizes[2]), 
					Catch::UnorderedEquals(vector<integer_t>{0, 2, 3, 4, 5, 6}));
			// CHECK_THAT(vector<integer_t>(table + 3*CAP, table + 3*CAP + table_sizes[3]), 
			// 		Catch::UnorderedEquals(vector<integer_t>{0, 1, 2, 3, 4, 5, 6}));
			break;
		case 3:
			CHECK_THAT(vector<integer_t>(table + 0*CAP, table + 0*CAP + table_sizes[0]), 
					Catch::UnorderedEquals(vector<integer_t>{0}));
			CHECK_THAT(vector<integer_t>(table + 1*CAP, table + 1*CAP + table_sizes[1]), 
					Catch::UnorderedEquals(vector<integer_t>{1, 2}));
			CHECK_THAT(vector<integer_t>(table + 2*CAP, table + 2*CAP + table_sizes[2]), 
					Catch::UnorderedEquals(vector<integer_t>{0, 2, 3, 4, 5, 6}));
			// CHECK_THAT(vector<integer_t>(table + 3*CAP, table + 3*CAP + table_sizes[3]), 
			// 		Catch::UnorderedEquals(vector<integer_t>{0, 1, 2, 3, 4, 5, 6}));
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

	attach();

#ifdef 0
	CrossMap cm(N, N-1, proc_num);

	for (int i=0; i<N; i++) {
		if (i < proc_rank) {
			cm._idx2index[i] = i;
		} else if (i ==  proc_rank){
			cm._idx2index[i] = -1;
		} else {
			cm._idx2index[i] = i - 1;
		}
	}

	switch (proc_rank) {
		case 0:
			for (int s=0; s<N; s++) {
				if (s == proc_rank) {
					continue;
				} else if (s < proc_rank) {
					cm._index2ridx[s*proc_num+s] = N + proc_rank - 1;
				} else {
					cm._index2ridx[(s-1)*proc_num+s] = N + proc_rank;
				}
			}
			break;
		case 1:
			for (int s=0; s<N; s++) {
				if (s == proc_rank) {
					continue;
				} else if (s < proc_rank) {
					cm._index2ridx[s*proc_num+s] = N + proc_rank - 1;
				} else {
					cm._index2ridx[(s-1)*proc_num+s] = N + proc_rank;
				}
			}
			break;
		case 2:
			for (int s=0; s<N; s++) {
				if (s == proc_rank) {
					continue;
				} else if (s < proc_rank) {
					cm._index2ridx[s*proc_num+s] = N + proc_rank - 1;
				} else {
					cm._index2ridx[(s-1)*proc_num+s] = N + proc_rank;
				}
			}
			break;
		case 3:
			for (int s=0; s<N; s++) {
				if (s == proc_rank) {
					continue;
				} else if (s < proc_rank) {
					cm._index2ridx[s*proc_num+s] = N + proc_rank - 1;
				} else {
					cm._index2ridx[(s-1)*proc_num+s] = N + proc_rank;
				}
			}
			break;
		default:
			printf("Test case should carry out on four processes\n");
			exit(-1);
			break;
	}

	CrossSpike cs(proc_rank, proc_num * THREAD_NUM, DELAY, 1);
	cs._recv_offset[0] = 0;
	cs._send_offset[0] = 0;

	for (int i=0; i<proc_num; i++) {
		cs._recv_offset[i+1] = cs._recv_offset[i] + DELAY * N;
		cs._send_offset[i+1] = cs._send_offset[i] + DELAY * N;
	}

	cs.alloc();

	char name[1024];
	char name_t[1024];
	sprintf(name, "%s_%d.map", argv[0], proc_rank);
	sprintf(name_t, "%s_%d", argv[0], proc_rank);

	cm.log(name);
	cm.to_gpu();

	cs.to_gpu();

	nid_t *table_gpu = TOGPU(table,  (DELAY+1) * CAP);
	nid_t *table_sizes_gpu = TOGPU(table_sizes, DELAY+1);

	for (int t=0; t<DELAY; t++) {
		cs.fetch_gpu(&cm, (nid_t *)table_gpu, (nsize_t *)table_sizes_gpu, CAP, proc_num, DELAY, t, 2, 32);
		cs.update_gpu(t);
		cs.log_gpu(t, name_t); 
		cs.upload_gpu((nid_t *)table_gpu, (nsize_t *)table_sizes_gpu, (nsize_t *)table_sizes, CAP, DELAY, t, 2, 32);
	}


	COPYFROMGPU(table, table_gpu, (DELAY+1) * CAP);
	COPYFROMGPU(table_sizes, table_sizes_gpu, DELAY+1);

	MPI_Barrier(MPI_COMM_WORLD);

	for (int i=0; i<DELAY+1; i++) {
		printf("Rank %d:%d :", proc_rank, table_sizes[i]);
		for (int j=0; j<table_sizes[i]; j++) {
			printf("%d ", table[j + i * CAP]);
		}
		printf("\n");
	}
#endif


	int result = Catch::Session().run(argc, argv);
	MPI_Barrier(MPI_COMM_WORLD);

	MPI_Finalize();

	return result;
}
