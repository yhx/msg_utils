
#include "../helper/helper_c.h"
#include "../helper/helper_gpu.h"
#include "ProcBuf.h"

ProcBuf::ProcBuf(CrossSpike **cs, int proc_rank, int proc_num, int thread_num, int min_delay)
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

	_recv_offset = malloc_c<integer_t>(_proc_num);
	_send_offset = malloc_c<integer_t>(_proc_num);
	_recv_offset[0] = 0;
	_send_offset[0] = 0;
	for (int p=0; p<_proc_num-1; p++) {
		_recv_offset[p+1] = _recv_offset[p];
		_send_offset[p+1] = _send_offset[p];
		for (int t=0; t<_thread_num; t++) {
			int idx = p * _thread_num + t;
			for (int k=0; k<_thread_num; k++) {
				_recv_offset[p+1] += cs[k]->_recv_offset[idx+1] - cs[k]->_recv_offset[idx];
				_send_offset[p+1] += cs[k]->_send_offset[idx+1] - cs[k]->_send_offset[idx];
			}
		}
	}

	_rdata_offset = malloc_c<integer_t>(_proc_num * _thread_num);
	_sdata_offset = malloc_c<integer_t>(_proc_num * _thread_num);
	for (int p=0; p<_proc_num; p++) {
		// _rdata_offset[p*_thread_num] = _recv_offset[p];
		_sdata_offset[p*_thread_num] = _send_offset[p];
	}


	int size = _proc_num * _thread_num * (_min_delay +  1);

	_recv_start = malloc_c<integer_t>(size);
	_send_start = malloc_c<integer_t>(size);

	_recv_num = malloc_c<integer_t>(_proc_num);
	_send_num = malloc_c<integer_t>(_proc_num);

	_recv_data = malloc_c<nid_t>(_rdata_size[thread_num+1]);
	_send_data = malloc_c<nid_t>(_sdata_size[thread_num+1]);
}

ProcBuf::~ProcBuf()
{
	_cs = NULL;

	_rdata_size = free_c(_rdata_size);
	_sdata_size = free_c(_sdata_size);

	_recv_start = free_c(_recv_start);
	_send_start = free_c(_send_start);

	_recv_offset = free_c(_recv_offset);
	_send_offset = free_c(_send_offset);

	_rdata_offset = free_c(_recv_offset);
	_sdata_offset = free_c(_send_offset);

	_recv_num = free_c(_recv_num);
	_send_num = free_c(_send_num);

	_recv_data = free_c(_recv_data);
	_send_data = free_c(_send_data);
}

int ProcBuf::update_gpu(const int &thread_id, const int &time, pthread_barrier_t *barrier)
{
	int curr_delay = time % _min_delay;
	if (curr_delay >= _min_delay - 1) {
		// msg start info
		CrossSpike *cst = _cs[thread_id];
		COPYFROMGPU(cst->_send_start, cst->_gpu_array->_send_start, cst->_proc_num * (cst->_min_delay+1));

		pthread_barrier_wait(barrier);
		if (thread_id == 0) {
			for (int i=0; i<_thread_num; i++) {
				int size = _thread_num * (_min_delay+1);
				MPI_Alltoall(_cs[i]->_send_start, size, MPI_INTEGER_T, _recv_start + i * size, size, MPI_INTEGER_T, MPI_COMM_WORLD);
			}
		}
		// calc thread offset
		int bk_size = _proc_num / _thread_num;
		for (int p=0; p<bk_size; p++) {
			int pid = bk_size * thread_id + p;
			for (int t=0; t<_thread_num-1; t++) {
				int idx = pid * _thread_num + t;
				_sdata_offset[idx+1] = _sdata_offset[idx] + (_cs[thread_id]->_send_start[idx*(_min_delay+1) + _min_delay] - _cs[thread_id]->_send_start[idx * (_min_delay+1)]);
			}
			int idx = pid * _thread_num + _thread_num - 1;
			_send_num[pid] = _sdata_offset[idx] + (_cs[_thread_num-1]->_send_start[idx*(_min_delay+1) + _min_delay] - _cs[_thread_num-1]->_send_start[idx * (_min_delay+1)]);
		}

		// msg thread offset
		if (thread_id == 0) {
			for (int i=0; i<_thread_num; i++) {
				MPI_Alltoall(_sdata_offset, _thread_num, MPI_INTEGER_T, _sdata_offset, _thread_num, MPI_INTEGER_T, MPI_COMM_WORLD);
			}
		}
		// fetch data
		for (int p=0; p<_proc_num; p++) {
			for (int t=0; t<_thread_num; t++) {
				int idx = p * _thread_num + t;
				COPYFROMGPU(_send_data + _sdata_offset[idx], cst->_gpu_array->_send_data + cst->_send_start[idx*(_min_delay+1)], cst->_send_start[idx*(_min_delay+1) + _min_delay] - cst->_send_start[idx * (_min_delay+1)]);
			}
		}
		// msg data
		int ret = MPI_Alltoallv(_send_data, _send_num, _send_offset, MPI_NID_T, _recv_data, _recv_num, _recv_offset, MPI_INTEGER_T, MPI_COMM_WORLD);
		assert(ret == MPI_SUCCESS);
	} else {
		_cs[thread_id]->update_gpu(time);
	}

	return 0;
}
