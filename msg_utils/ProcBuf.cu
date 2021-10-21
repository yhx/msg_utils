
#include "../helper/helper_c.h"
#include "../helper/helper_gpu.h"
#include "ProcBuf.h"


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
				MPI_Alltoall(_sdata_offset, _thread_num, MPI_INTEGER_T, _rdata_offset, _thread_num, MPI_INTEGER_T, MPI_COMM_WORLD);
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
		if (thread_id == 0) {
			int ret = MPI_Alltoallv(_send_data, _send_num, _send_offset, MPI_NID_T, _recv_data, _recv_num, _recv_offset, MPI_INTEGER_T, MPI_COMM_WORLD);
			assert(ret == MPI_SUCCESS);
		}
	} else {
		_cs[thread_id]->update_gpu(time);
	}

	return 0;
}

int ProcBuf::upload_gpu(const int &thread_id, nid_t *tables, nsize_t *table_sizes, nsize_t *c_table_sizes, const size_t &table_cap, const int &max_delay, const int &time, const int &grid, const int &block)
{
	int curr_delay = time % _min_delay;
	if (curr_delay >= _min_delay -1) {
#ifdef PROF
		ts = MPI_Wtime();
#endif
		COPYFROMGPU(c_table_sizes, table_sizes, max_delay+1);
#ifdef PROF
		te = MPI_Wtime();
		_gpu_wait += te - ts;
#endif

#if 0
// #ifdef ASYNC
#ifdef PROF
		ts = MPI_Wtime();
#endif 
		MPI_Status status_t;
		int ret = MPI_Wait(&_request, &status_t);
		assert(ret == MPI_SUCCESS);
#ifdef PROF
		te = MPI_Wtime();
		_cpu_wait_gpu += te - ts;
#endif
#endif

		for (int d=0; d < _min_delay; d++) {
			int delay_idx = (time-_min_delay+2+d+max_delay)%(max_delay+1);
			for (int p = 0; p<_proc_num; p++) {
				for (int t = 0; t<_thread_num; t++) {
					int idx = p * _thread_num + t;
					int start = _recv_start[idx*(_min_delay+1)+d];
					int end = _recv_start[idx*(_min_delay+1)+d+1];
					int num = end - start;
					if (num > 0) {
						assert(c_table_sizes[delay_idx] + num <= table_cap);
						COPYTOGPU(tables + table_cap*delay_idx + c_table_sizes[delay_idx], _recv_data + _recv_offset[p] + _rdata_offset[thread_id] + start, num);
						c_table_sizes[delay_idx] += num;
					}
				}
			}
		}
		COPYTOGPU(table_sizes, c_table_sizes, max_delay+1);

		{ // Reset
			gpuMemset(_cs[thread_id]->_gpu_array->_recv_start, 0, _min_delay * _proc_num + _proc_num);
			gpuMemset(_cs[thread_id]->_gpu_array->_send_start, 0, _min_delay * _proc_num + _proc_num);

			memset_c(_recv_num, 0, _proc_num);
			memset_c(_send_num, 0, _proc_num);
		}
	}

	return 0;
}
