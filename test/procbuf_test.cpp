
#define CATCH_CONFIG_RUNNER

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <pthread.h>

#include "catch.hpp"

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
const int UPDATE_DELAY = 1;
const int UPLOAD_DELAY = 3;
const int THREAD_NUM = 2;

int proc_rank = -1;
int proc_num = -1;


struct ThreadPara {
	ProcBuf *pbuf;
	pthread_barrier_t *barrier;
	int tid;
};

inline nid_t get_value(int d, int s_p, int s_t,  int d_p, int d_t, int d_n) {
	return 100000* d + 10000 * s_p + 1000 * s_t + 100 * d_p + 10 * d_t + d_n;
}

#if 0
void * check_update_1(void *para) {
	ThreadPara *tmp = static_cast<ThreadPara*>(para);
	int tid = tmp->tid;

	CrossSpike &cs = *(tmp->pbuf->_cs[tid]);

	for (int p=0; p<proc_num; p++) {
		for (int t=0; t<THREAD_NUM; t++) {
			int idx = p * THREAD_NUM + t;
			cs._recv_start[idx*(UPDATE_DELAY+1)+0] = 0;
			cs._send_start[idx*(UPDATE_DELAY+1)+0] = 0;
			for (int d=0; d<UPDATE_DELAY; d++) {
				int data_size = idx;
				cs._send_start[idx*(UPDATE_DELAY+1)+d+1] = cs._send_start[idx*(UPDATE_DELAY+1)+d] + data_size;
				for (int k=0; k<data_size; k++) {
					cs._send_data[cs._send_offset[idx] + cs._send_start[idx*(UPDATE_DELAY+1)+d] + k] = get_value(d, proc_rank, tid, p, t, k); 
				}
			}
		}

	}

	pthread_barrier_wait(tmp->barrier);

	tmp->pbuf->update_cpu(tid, UPDATE_DELAY-1, tmp->barrier);

	return 0;
}

TEST_CASE("CHECK update 1", "") {
	CrossSpike **css = new CrossSpike*[THREAD_NUM];
	for (int tid=0; tid<THREAD_NUM; tid++) {
		css[tid] = new CrossSpike(proc_rank, proc_num * THREAD_NUM, UPDATE_DELAY, 0);
		css[tid]->_recv_offset[0] = 0;
		css[tid]->_send_offset[0] = 0;

		for (int i=0; i<proc_num*THREAD_NUM; i++) {
			css[tid]->_recv_offset[i+1] = css[tid]->_recv_offset[i] + proc_num*THREAD_NUM;
			css[tid]->_send_offset[i+1] = css[tid]->_send_offset[i] + proc_num*THREAD_NUM;
		}

		css[tid]->alloc();
	}

	ProcBuf pbuf(css, proc_rank, proc_num, THREAD_NUM, UPDATE_DELAY);

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
				for (int d=0; d<UPDATE_DELAY; d++) {
					int idx = p * THREAD_NUM + d_t;
					int start = pbuf._recv_start[s_t*THREAD_NUM*proc_num*(UPDATE_DELAY+1)+idx*(UPDATE_DELAY+1)+d];
					int end = pbuf._recv_start[s_t*THREAD_NUM*proc_num*(UPDATE_DELAY+1)+idx*(UPDATE_DELAY+1)+d+1];
					REQUIRE(end-start == proc_rank * THREAD_NUM + d_t);
					for (int i=start; i<end; i++) {
						REQUIRE(pbuf._recv_data[pbuf._recv_offset[p]+pbuf._rdata_offset[idx]+i+count] == get_value(d, p, s_t, proc_rank, d_t, i-start));
					}
					count += end - start;
				}
			}
		}
	}


	// CHECK_THAT(vector<nid_t>(table + 0*CAP, table + 0*CAP + table_sizes[0]), Catch::UnorderedEquals(vector<nid_t>{0, 4, 5, 6}));
}


void * check_update_2(void *para) {
	ThreadPara *tmp = static_cast<ThreadPara*>(para);
	int tid = tmp->tid;

	CrossSpike &cs = *(tmp->pbuf->_cs[tid]);

	for (int p=0; p<proc_num; p++) {
		for (int t=0; t<THREAD_NUM; t++) {
			int idx = p * THREAD_NUM + t;
			cs._recv_start[idx*(UPDATE_DELAY+1)+0] = 0;
			cs._send_start[idx*(UPDATE_DELAY+1)+0] = 0;
			for (int d=0; d<UPDATE_DELAY; d++) {
				int data_size = THREAD_NUM*proc_num+1-(proc_rank*THREAD_NUM+tid);
				cs._send_start[idx*(UPDATE_DELAY+1)+d+1] = cs._send_start[idx*(UPDATE_DELAY+1)+d] + data_size;
				for (int k=0; k<data_size; k++) {
					cs._send_data[cs._send_offset[idx] + cs._send_start[idx*(UPDATE_DELAY+1)+d] + k] = get_value(d, proc_rank, tid, p, t, k); 
				}
			}
		}

	}

	pthread_barrier_wait(tmp->barrier);

	tmp->pbuf->update_cpu(tid, 1, tmp->barrier);

	return 0;
}

TEST_CASE("CHECK update 2", "") {
	CrossSpike **css = new CrossSpike*[THREAD_NUM];
	for (int tid=0; tid<THREAD_NUM; tid++) {
		css[tid] = new CrossSpike(proc_rank, proc_num * THREAD_NUM, UPDATE_DELAY, 0);
		css[tid]->_recv_offset[0] = 0;
		css[tid]->_send_offset[0] = 0;

		for (int i=0; i<proc_num*THREAD_NUM; i++) {
			css[tid]->_recv_offset[i+1] = css[tid]->_recv_offset[i] + proc_num*THREAD_NUM+1;
			css[tid]->_send_offset[i+1] = css[tid]->_send_offset[i] + proc_num*THREAD_NUM+1;
		}

		css[tid]->alloc();
	}

	ProcBuf pbuf(css, proc_rank, proc_num, THREAD_NUM, UPDATE_DELAY);

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
				for (int d=0; d<UPDATE_DELAY; d++) {
					int idx = p * THREAD_NUM + d_t;
					int start = pbuf._recv_start[s_t*THREAD_NUM*proc_num*(UPDATE_DELAY+1)+idx*(UPDATE_DELAY+1)+d];
					int end = pbuf._recv_start[s_t*THREAD_NUM*proc_num*(UPDATE_DELAY+1)+idx*(UPDATE_DELAY+1)+d+1];
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
#endif

#if 1
void * check_upload1(void *para) {
	const int LOCAL = 4;

	ThreadPara *tmp = static_cast<ThreadPara*>(para);
	int tid = tmp->tid;
	int own_id = proc_rank * THREAD_NUM + tid;

	int num = proc_num * THREAD_NUM;
	int n_num = LOCAL + num;
	int cross_num = num - 1;
	int n_cap = n_num + cross_num;

	CrossMap cm(n_num, cross_num, proc_num*THREAD_NUM);

	for (int n=0; n<n_num; n++) {
		if (n < LOCAL) {
			cm._idx2index[n] = -1;
		} else if (n < LOCAL+own_id) {
			cm._idx2index[n] = n-LOCAL;
		} else if (n == LOCAL+own_id) {
			cm._idx2index[n] = -1;
		} else {
			cm._idx2index[n] = n - LOCAL - 1;
		}
	}

	for (int s=0; s<num-1; s++) {
		for (int d=0; d<num; d++) {
			if (d == own_id) {
				cm._index2ridx[s*num+d] = -1;
			} else if (d < own_id) {
				cm._index2ridx[s*num+d] = get_value(0, proc_rank, tid, d/THREAD_NUM, d%THREAD_NUM, s);
			} else {
				cm._index2ridx[s*num+d] = get_value(0, proc_rank, tid, d/THREAD_NUM, d%THREAD_NUM, s);
			}
		}
	}

	CrossSpike &cs = *(tmp->pbuf->_cs[tid]);

	char name[1024];
	char name_t[1024];
	sprintf(name, "%s_%d.map", "upload_gpu_test", own_id);
	sprintf(name_t, "%s_%d", "upload_gpu_test", own_id);

	cm.log(name);

	nid_t *table = malloc_c<nid_t>((UPLOAD_DELAY+1) * n_cap); 

	for (int i=0; i<LOCAL; i++) {
		table[i] = i;
		table[n_cap+i] = i;
		table[3*n_cap+i] = i;
	}

	for (int i=0; i<2; i++) {
		table[n_cap+LOCAL+i] = LOCAL+i;
		table[3*n_cap+LOCAL+i] = LOCAL+i;
	}

	for (int i=0; i<2; i++) {
		table[3*n_cap+LOCAL+2+i] = LOCAL + num-1-i;
	}

	for (int i=0; i<n_num; i++) {
		table[2*n_cap + i] = i;
	}

	nsize_t table_sizes[UPLOAD_DELAY+1] = {LOCAL, LOCAL+2, n_num, LOCAL+4};


	for (int t=0; t<UPLOAD_DELAY; t++) {
		cs.fetch_cpu(&cm, (nid_t *)table, (nsize_t *)table_sizes, n_cap, proc_num, UPLOAD_DELAY, t);
		pthread_barrier_wait(tmp->barrier);
		tmp->pbuf->update_cpu(tid, t, tmp->barrier);
		cs.log_cpu(t, name_t); 
		tmp->pbuf->upload_cpu(tid, (nid_t *)table, (nsize_t *)table_sizes, n_cap, UPLOAD_DELAY, t);
	}

	CHECK_THAT(vector<nid_t>(table + 0*n_cap, table + 0*n_cap + table_sizes[0]), 
			Catch::UnorderedEquals(vector<nid_t>{0, 1, 2, 3}));

	vector<nid_t> res1 = {0, 1, 2, 3, 4, 5};
	for (int s=0; s<num; s++) {
		if (own_id == 0 && s != own_id) {
			res1.push_back(get_value(0, s/THREAD_NUM, s%THREAD_NUM, proc_rank, tid, 0));
		} else if (own_id == 1 && s != own_id) {
			if (s < own_id) {
				res1.push_back(get_value(0, s/THREAD_NUM, s%THREAD_NUM, proc_rank, tid, own_id-1));
			} else {
				res1.push_back(get_value(0, s/THREAD_NUM, s%THREAD_NUM, proc_rank, tid, own_id));
			}
		} else {
			continue;
		}
	}
	CHECK_THAT(vector<nid_t>(table + 1*n_cap, table + 1*n_cap + table_sizes[1]), 
			Catch::UnorderedEquals(res1));

	vector<nid_t> res2;
	for (int i=0; i<n_num; i++) {
		res2.push_back(i);
	}

	for (int s=0; s<num; s++) {
		if (s == own_id) {
			continue;
		} else if (s < own_id) {
				res2.push_back(get_value(0, s/THREAD_NUM, s%THREAD_NUM, proc_rank, tid, own_id-1));
		} else {
				res2.push_back(get_value(0, s/THREAD_NUM, s%THREAD_NUM, proc_rank, tid, own_id));
		}
	}
	CHECK_THAT(vector<nid_t>(table + 2*n_cap, table + 2*n_cap + table_sizes[2]), 
			Catch::UnorderedEquals(res2));

	return 0;
}

TEST_CASE("CHECK upload", "") {
	CrossSpike **css = new CrossSpike*[THREAD_NUM];
	for (int tid=0; tid<THREAD_NUM; tid++) {
		css[tid] = new CrossSpike(proc_rank, proc_num * THREAD_NUM, UPLOAD_DELAY, 0);
		css[tid]->_recv_offset[0] = 0;
		css[tid]->_send_offset[0] = 0;

		for (int i=0; i<proc_num*THREAD_NUM; i++) {
			css[tid]->_recv_offset[i+1] = css[tid]->_recv_offset[i] + proc_num*THREAD_NUM * UPLOAD_DELAY;
			css[tid]->_send_offset[i+1] = css[tid]->_send_offset[i] + proc_num*THREAD_NUM * UPLOAD_DELAY;
		}

		css[tid]->alloc();
	}

	ProcBuf pbuf(css, proc_rank, proc_num, THREAD_NUM, UPLOAD_DELAY);

	pthread_barrier_t g_proc_barrier;

	pthread_barrier_init(&g_proc_barrier, NULL, THREAD_NUM);
	pthread_t *thread_ids = malloc_c<pthread_t>(THREAD_NUM);
	assert(thread_ids != NULL);

	ThreadPara *para = new ThreadPara[2];;

	for (int i=0; i<THREAD_NUM; i++) {
		para[i].pbuf = &pbuf;
		para[i].barrier = &g_proc_barrier;
		para[i].tid = i;
		int ret = pthread_create(&(thread_ids[i]), NULL, &check_upload1, (void*)(&para[i]));
		assert(ret == 0);
	}

	for (int i=0; i<THREAD_NUM; i++) {
		pthread_join(thread_ids[i], NULL);
	}
	pthread_barrier_destroy(&g_proc_barrier);

}
#endif

int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &proc_num);

	to_attach();

	int result = Catch::Session().run(argc, argv);
	MPI_Barrier(MPI_COMM_WORLD);

	MPI_Finalize();

	return result;
}
