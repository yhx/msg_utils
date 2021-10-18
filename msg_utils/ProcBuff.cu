
#include "../helper/helper_c.h"
#include "../helper/helper_gpu.h"
#include "ProcBuff.h"

ProcBuff::ProcBuff(CrossSpike **cs, int proc_rank, int proc_num, int thread_num, int min_delay)
{
	_cs = cs;
	_proc_rank = proc_rank;
	_proc_num = proc_num;
	_thread_num = thread_num;
	_min_delay = min_delay;

	_rdata_size = malloc_c<size_t>(_thread_num + 1);
	_sdata_size = malloc_c<size_t>(_thread_num + 1);

	_rdata_size[0] = 0;
	_sdata_size[0] = 0;

	for (int i=0; i<thread_num; i++) {
		_rdata_size[i+1] = _rdata_size[i] + cs[i]->_recv_offset[cs[i]->_proc_num];
		_sdata_size[i+1] = _sdata_size[i] + cs[i]->_send_offset[cs[i]->_proc_num];
	}

	_recv_offset = malloc_c<integer_t>(_proc_num * _thread_num);
	_send_offset = malloc_c<integer_t>(_proc_num * _thread_num);
	// _recv_offset = malloc_c<integer_t>(_proc_num);
	// _send_offset = malloc_c<integer_t>(_proc_num);
	_recv_offset[0] = 0;
	_send_offset[0] = 0;
	for (int p=0; p<_proc_num; p++) {
		_recv_offset[(p+1)*_thread_num] = _recv_offset[p*_thread_num];
		_send_offset[(p+1)*_thread_num] = _send_offset[p*_thread_num];
		for (int t=0; t<_thread_num; t++) {
			int idx = p * _thread_num + t;
			for (int k=0; k<_thread_num; k++) {
				_recv_offset[(p+1)*thread_num] += cs[k]->_recv_offset[idx+1] - cs[k]->_recv_offset[idx];
				_send_offset[(p+1)*thread_num] += cs[k]->_send_offset[idx+1] - cs[k]->_send_offset[idx];
			}
		}
	}

	int size = _proc_num * _thread_num * (_min_delay +  1);

	_recv_start = malloc_c<integer_t>(size);
	_send_start = malloc_c<integer_t>(size);

	_recv_num = malloc_c<integer_t>(_proc_num);
	_send_num = malloc_c<integer_t>(_proc_num);

	_recv_data = malloc_c<nid_t>(_rdata_size[thread_num+1]);
	_send_data = malloc_c<nid_t>(_sdata_size[thread_num+1]);
}

ProcBuff::~ProcBuff()
{
	_cs = NULL;

	_rdata_size = free_c(_rdata_size);
	_sdata_size = free_c(_sdata_size);

	_recv_start = free_c(_recv_start);
	_send_start = free_c(_send_start);

	_recv_offset = free_c(_recv_offset);
	_send_offset = free_c(_send_offset);
	_recv_num = free_c(_recv_num);
	_send_num = free_c(_send_num);

	_recv_data = free_c(_recv_data);
	_send_data = free_c(_send_data);
}

int ProcBuff::update_gpu(const int &thread_id, const int &time, pthread_barrier_t *barrier)
{
	int curr_delay = time % _min_delay;
	if (curr_delay >= _min_delay - 1) {
		// msg start info
		CrossSpike *p = _cs[thread_id];
		COPYFROMGPU(p->_send_start, p->_gpu_array->_send_start, p->_proc_num * (p->_min_delay+1));
		
		pthread_barrier_wait(barrier);
		if (thread_id == 0) {
			for (int i=0; i<_thread_num; i++) {
				int size = _thread_num * (_min_delay+1);
				MPI_Alltoall(_cs[i]->_send_start, size, MPI_INTEGER_T, _recv_start + i * size, size, MPI_INTEGER_T, MPI_COMM_WORLD);
			}
		}
		// msg thread offset
		// msg data
	} else {
		_cs[thread_id]->update_gpu(time);
	}

	return 0;
}
