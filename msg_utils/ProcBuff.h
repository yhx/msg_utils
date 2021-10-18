#ifndef PROCBUFF_H
#define PROCBUFF_H

#include "CrossSpike.h"

class ProcBuff {
public:
	ProcBuff(CrossSpike **cs, int proc_rank, int proc_num, int thread_num, int min_delay);
	~ProcBuff();

	int update_gpu(const int &thread_id, const int &time, pthread_barrier_t *barrier);

	// Cap _proc_num * _thread_num * (_min_delay + 1)
	integer_t *_recv_start;
	integer_t *_send_start;

	// Cap _proc_num + 1
	integer_t *_recv_offset;
	integer_t *_send_offset;

	// Cap _proc_num
	integer_t *_recv_num;
	integer_t *_send_num;

	// Cap _rdata_size[_thread_num]
	nid_t *_recv_data;
	// Cap _sdata_size[_thread_num]
	nid_t *_send_data;

	// Cap _thread_num
	CrossSpike **_cs;

	// Cap _thread_num + 1;
	size_t *_rdata_size;
	size_t *_sdata_size;

	int _proc_rank;
	int _proc_num;
	int _thread_num;
	int _min_delay;
};

#endif // PROCBUFF_H
