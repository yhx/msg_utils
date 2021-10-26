
#define CATCH_CONFIG_RUNNER

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <pthread.h>

#include "catch.hpp"

// #include "../helper/helper_gpu.h"
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
const int DELAY = 1;
// const int N = 4;
// const uinteger_t CAP = 8;
const int THREAD_NUM = 2;

int proc_rank = -1;
int proc_num = -1;

// nid_t table[(DELAY+1) * CAP] = {
// 	0, 0, 0, 0, 0, 0, 0, 0,
// 	1, 2, 0, 0, 0, 0, 0, 0,
// 	2, 3, 0, 0, 0, 0, 0, 0,
// 	1, 2, 3, 4, 0, 0, 0, 0
// };
// 
// nsize_t table_sizes[DELAY+1] = {1, 2, 3, 4};

struct ThreadPara {
	ProcBuf *pbuf;
	pthread_barrier_t *barrier;
	int tid;
};

inline int get_value(int d, int s_p, int s_t,  int d_p, int d_t, int d_n) {
	return 100000* d + 10000 * s_p + 1000 * s_t + 100 * d_p + 10 * d_t + d_n;
}

void * check_update_1(void *para) {
	ThreadPara *tmp = static_cast<ThreadPara*>(para);
	int tid = tmp->tid;

	CrossSpike &cs = *(tmp->pbuf->_cs[tid]);

	for (int p=0; p<proc_num; p++) {
		for (int t=0; t<THREAD_NUM; t++) {
			int idx = p * THREAD_NUM + t;
			cs._recv_start[idx*(DELAY+1)+0] = 0;
			cs._send_start[idx*(DELAY+1)+0] = 0;
			for (int d=0; d<DELAY; d++) {
				int data_size = idx;
				cs._send_start[idx*(DELAY+1)+d+1] = cs._send_start[idx*(DELAY+1)+d] + data_size;
				for (int k=0; k<data_size; k++) {
					cs._send_data[cs._send_offset[idx] + cs._send_start[idx*(DELAY+1)+d] + k] = get_value(d, proc_rank, tid, p, t, k); 
				}
			}
		}

	}

	pthread_barrier_wait(tmp->barrier);

	tmp->pbuf->update_cpu(tid, 1, tmp->barrier);

	return 0;
}

TEST_CASE("CHECK Update 1", "") {
	CrossSpike **css = new CrossSpike*[THREAD_NUM];
	for (int tid=0; tid<THREAD_NUM; tid++) {
		css[tid] = new CrossSpike(proc_rank, proc_num * THREAD_NUM, DELAY, 0);
		css[tid]->_recv_offset[0] = 0;
		css[tid]->_send_offset[0] = 0;

		for (int i=0; i<proc_num*THREAD_NUM; i++) {
			css[tid]->_recv_offset[i+1] = css[tid]->_recv_offset[i] + proc_num*THREAD_NUM;
			css[tid]->_send_offset[i+1] = css[tid]->_send_offset[i] + proc_num*THREAD_NUM;
		}

		css[tid]->alloc();
	}

	ProcBuf pbuf(css, proc_rank, proc_num, THREAD_NUM, DELAY);

	pthread_barrier_t g_proc_barrier;

	pthread_barrier_init(&g_proc_barrier, NULL, THREAD_NUM);
	pthread_t *thread_ids = malloc_c<pthread_t>(THREAD_NUM);
	assert(thread_ids != NULL);

	ThreadPara *para = new ThreadPara[2];;

	for (int i=0; i<THREAD_NUM; i++) {
		para[i].pbuf = &pbuf;
		para[i].barrier = &g_proc_barrier;
		para[i].tid = i;
		int ret = pthread_create(&(thread_ids[i]), NULL, &check_update_1, (void*)(&para[i]));
		assert(ret == 0);
	}

	for (int i=0; i<THREAD_NUM; i++) {
		pthread_join(thread_ids[i], NULL);
	}
	pthread_barrier_destroy(&g_proc_barrier);

	for (int p=0; p<proc_num; p++) {
		for (int d_t=0; d_t<THREAD_NUM; d_t++) {
			int count = 0;
			for (int s_t=0; s_t<THREAD_NUM; s_t++) {
				for (int d=0; d<DELAY; d++) {
					int idx = p * THREAD_NUM + d_t;
					int start = pbuf._recv_start[s_t*THREAD_NUM*proc_num*(DELAY+1)+idx*(DELAY+1)+d];
					int end = pbuf._recv_start[s_t*THREAD_NUM*proc_num*(DELAY+1)+idx*(DELAY+1)+d+1];
					REQUIRE(end-start == proc_rank * THREAD_NUM + d_t);
					for (int i=start; i<end; i++) {
						REQUIRE(pbuf._recv_data[pbuf._recv_offset[p]+pbuf._rdata_offset[idx]+i+count] == get_value(d, p, s_t, proc_rank, d_t, i-start));
					}
					count += end - start;
				}
			}
		}
	}


	// CHECK_THAT(vector<integer_t>(table + 0*CAP, table + 0*CAP + table_sizes[0]), Catch::UnorderedEquals(vector<integer_t>{0, 4, 5, 6}));
}

void * check_update_2(void *para) {
	ThreadPara *tmp = static_cast<ThreadPara*>(para);
	int tid = tmp->tid;

	CrossSpike &cs = *(tmp->pbuf->_cs[tid]);

	for (int p=0; p<proc_num; p++) {
		for (int t=0; t<THREAD_NUM; t++) {
			int idx = p * THREAD_NUM + t;
			cs._recv_start[idx*(DELAY+1)+0] = 0;
			cs._send_start[idx*(DELAY+1)+0] = 0;
			for (int d=0; d<DELAY; d++) {
				int data_size = THREAD_NUM*proc_num+1-(proc_rank*THREAD_NUM+tid);
				cs._send_start[idx*(DELAY+1)+d+1] = cs._send_start[idx*(DELAY+1)+d] + data_size;
				for (int k=0; k<data_size; k++) {
					cs._send_data[cs._send_offset[idx] + cs._send_start[idx*(DELAY+1)+d] + k] = get_value(d, proc_rank, tid, p, t, k); 
				}
			}
		}

	}

	pthread_barrier_wait(tmp->barrier);

	tmp->pbuf->update_cpu(tid, 1, tmp->barrier);

	return 0;
}

TEST_CASE("CHECK Update 2", "") {
	CrossSpike **css = new CrossSpike*[THREAD_NUM];
	for (int tid=0; tid<THREAD_NUM; tid++) {
		css[tid] = new CrossSpike(proc_rank, proc_num * THREAD_NUM, DELAY, 0);
		css[tid]->_recv_offset[0] = 0;
		css[tid]->_send_offset[0] = 0;

		for (int i=0; i<proc_num*THREAD_NUM; i++) {
			css[tid]->_recv_offset[i+1] = css[tid]->_recv_offset[i] + proc_num*THREAD_NUM+1;
			css[tid]->_send_offset[i+1] = css[tid]->_send_offset[i] + proc_num*THREAD_NUM+1;
		}

		css[tid]->alloc();
	}

	ProcBuf pbuf(css, proc_rank, proc_num, THREAD_NUM, DELAY);

	pthread_barrier_t g_proc_barrier;

	pthread_barrier_init(&g_proc_barrier, NULL, THREAD_NUM);
	pthread_t *thread_ids = malloc_c<pthread_t>(THREAD_NUM);
	assert(thread_ids != NULL);

	ThreadPara *para = new ThreadPara[2];;

	for (int i=0; i<THREAD_NUM; i++) {
		para[i].pbuf = &pbuf;
		para[i].barrier = &g_proc_barrier;
		para[i].tid = i;
		int ret = pthread_create(&(thread_ids[i]), NULL, &check_update_2, (void*)(&para[i]));
		assert(ret == 0);
	}

	for (int i=0; i<THREAD_NUM; i++) {
		pthread_join(thread_ids[i], NULL);
	}
	pthread_barrier_destroy(&g_proc_barrier);

	for (int p=0; p<proc_num; p++) {
		for (int d_t=0; d_t<THREAD_NUM; d_t++) {
			int count = 0;
			for (int s_t=0; s_t<THREAD_NUM; s_t++) {
				for (int d=0; d<DELAY; d++) {
					int idx = p * THREAD_NUM + d_t;
					int start = pbuf._recv_start[s_t*THREAD_NUM*proc_num*(DELAY+1)+idx*(DELAY+1)+d];
					int end = pbuf._recv_start[s_t*THREAD_NUM*proc_num*(DELAY+1)+idx*(DELAY+1)+d+1];
					REQUIRE(end-start == THREAD_NUM*proc_num+1-(p * THREAD_NUM + s_t));
					for (int i=start; i<end; i++) {
						REQUIRE(pbuf._recv_data[pbuf._recv_offset[p]+pbuf._rdata_offset[idx]+i+count] == get_value(d, p, s_t, proc_rank, d_t, i-start));
					}
					count += end - start;
				}
			}
		}
	}

}

int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &proc_num);

	to_attach();

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
