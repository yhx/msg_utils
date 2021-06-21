
#include "helper/helper_c.h"
#include "helper/helper_gpu.h"
#include "CrossSpike.h"

CrossSpike::~CrossSpike()
{
	if (_proc_num > 0) {
		free_c(_recv_offset);
		free_c(_recv_start);
		free_c(_recv_num);
		free_c(_recv_data);

		free_c(_send_offset);
		free_c(_send_start);
		free_c(_send_num);
		free_c(_send_data);
	}

	if (_gpu_array) {
		gpuFree(_gpu_array->_recv_offset);
		gpuFree(_gpu_array->_recv_start);
		gpuFree(_gpu_array->_recv_num);
		gpuFree(_gpu_array->_recv_data);

		gpuFree(_gpu_array->_send_offset);
		gpuFree(_gpu_array->_send_start);
		gpuFree(_gpu_array->_send_num);
		gpuFree(_gpu_array->_send_data);

		_gpu_array->_proc_num = 0;
		_gpu_array->_min_delay = 0;
		_gpu_array->_gpu_array = NULL;

		delete _gpu_array;
	}

	_proc_num = 0;
	_min_delay = 0;
}

int CrossSpike::to_gpu()
{
	size_t size = _min_delay * _proc_num;
	size_t num_p_1 = _proc_num + 1;

	if (!_gpu_array) {
		_gpu_array = new CrossSpike;
		_gpu_array->_proc_num = _proc_num;
		_gpu_array->_min_delay = _min_delay;

		_gpu_array->_recv_offset = copyToGPU(_recv_offset, num_p_1);
		_gpu_array->_recv_start = copyToGPU(_recv_start, size+_proc_num);
		_gpu_array->_recv_num = copyToGPU(_recv_num, _proc_num);

		_gpu_array->_send_offset = copyToGPU(_send_offset, num_p_1);
		_gpu_array->_send_start = copyToGPU(_send_start, size+_proc_num);
		_gpu_array->_send_num = copyToGPU(_send_num, _proc_num);

		_gpu_array->_recv_data = copyToGPU(_recv_data, _recv_offset[_proc_num]);

		_gpu_array->_send_data = copyToGPU(_send_data, _send_offset[_proc_num]);
	} else {
		assert(_gpu_array->_proc_num == _proc_num);
		assert(_gpu_array->_min_delay == _min_delay);

		copyToGPU(_gpu_array->_recv_offset, _recv_offset, num_p_1);
		copyToGPU(_gpu_array->_recv_start, _recv_start, size+_proc_num);
		copyToGPU(_gpu_array->_recv_num, _recv_num, _proc_num);

		copyToGPU(_gpu_array->_send_offset, _send_offset, num_p_1);
		copyToGPU(_gpu_array->_send_start, _send_start, size+_proc_num);
		copyToGPU(_gpu_array->_send_num, _send_num, _proc_num);

		copyToGPU(_gpu_array->_recv_data, _recv_data, _recv_offset[_proc_num]);

		copyToGPU(_gpu_array->_send_data, _send_data, _send_offset[_proc_num]);
	}

	return 0;
}

template<typename TID, typename TSIZE>
__global__ void fetch_kernel(integer_t *data, integer_t *offset, integer_t *num, integer_t *idx2index, integer_t *index2ridx, TID *fired_table, TSIZE *fired_sizes, TSIZE fired_cap, int proc_num, int delay_idx, int min_delay, int delay)
{
	__shared__ integer_t cross_neuron_id[MAX_BLOCK_SIZE];
	__shared__ volatile integer_t cross_cnt;

	if (threadIdx.x == 0) {
		cross_cnt = 0;
	}
	__syncthreads();

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	T3 fired_size = fired_sizes[delay_idx];
	for (int node = 0; node < proc_num; node++) {
		for (int idx = tid; idx < fired_size; idx += blockDim.x * gridDim.x) {
			integer_t nid = static_cast<integer_t>(fired_table[fired_cap*delay_idx + idx]);
			T3 tmp = idx2index[nid];
			if (tmp >= 0) {
				T3 map_nid = index2ridx[tmp*proc_num + node];
				if (map_nid >= 0) {
					size_t test_loc = static_cast<size_t>(atomicAdd(&cross_cnt, 1));
					if (test_loc < MAX_BLOCK_SIZE) {
						cross_neuron_id[test_loc] = static_cast<T1>(map_nid);
					}
				}
			}
			__syncthreads();

			if (cross_cnt > 0) {
				int idx_t = node * (min_delay + 1) + delay + 1;
				merge2array(cross_neuron_id, cross_cnt, data + offset[node] + num[idx_t], &(num[idx_t]), static_cast<integer_t>(fired_cap*node));
				if (threadIdx.x == 0) {
					cross_cnt = 0;
				}
			}
			__syncthreads();
		}
		__syncthreads();
	}
}

template<typename TID, typename TSIZE>
int CrossSpike::fetch_gpu(const CrossMap *map, const TID *tables, const TSIZE *table_sizes, const TSIZE table_cap, const int &proc_num, const int &max_delay, const int &time, const int &grid, const int &block)
{
	int delay_idx = time % (max_delay + 1);
	int curr_delay = time % _min_delay;
	fetch_kernel<<<grid, block>>>(_gpu_array->_send_data, _gpu_array->send_offset, _gpu_array->_send_start, map->_idx2index, map->_crossnodeIndex2idx, tables, table_sizes, table_cap, proc_num, delay_idx, _min_delay, curr_delay)
}

__global__ void update_kernel(integer_t *start, int proc_num, int min_delay, int curr_delay)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i=tid; i<proc_num; i++) {
		start[i*(min_delay+1)+curr_delay+2] = start[i*(min_delay+1)+curr_delay+1];
	}
}

int CrossSpike::update_gpu(const int &curr_delay)
{
	if (curr_delay > _min_delay -1) {
		if (_proc_num > _gpu_num) {
			copyFromGPU(_send_start, _gpu_array->_send_start, _proc_num * (_min_delay + 1));
			copyFromGPU(_send_data, _gpu_array->_send_data, _send_offset[_proc_num]);
		}
		msg_gpu();
	} else {
		cudaDeviceSynchronize();
		update_kernel<<<1, _proc_num>>>(_gpu_array->_send_start, _proc_num, _min_delay, curr_delay);
	}

	return 0;
}

int CrossSpike::msg_gpu(ncclComm_t &comm_gpu)
{
	for (int i=0; i<_proc_num; i++) {
		if (i/_gpu_num == _gpu_group) {
			_send_num[i] = 0;
		} else {
			_send_num[i] = _send_start[i*(_min_delay+1)+_min_delay];
		}
	}

	// int num_size = _min_delay * _proc_num;
	// print_mpi_x32(_send_num, num_size, "Send Num");
	// print_mpi_x32(_recv_num, num_size, "To Recv Num");

	cudaDeviceSynchronize();
	ncclGroupStart();
	int size = _min_delay + 1;
	int r_offset = _gpu_group * _gpu_num;
	for (int r=0; r<_gpu_num; r++) {
		if (r != _gpu_rank) {
			ncclSend(_gpu_array->_send_start + ((r_offset + r)*size), size, NCCL_INTEGER_T, r, comm_gpu, s);
			ncclRecv(_gpu_array->_recv_start + ((r_offset + r)*size), size, NCCL_INTEGER_T, r, comm_gpu, s);
		}
	}
	ncclGroupEnd();


	MPI_Alltoall(_send_start, _min_delay+1, MPI_INTEGER_T, _recv_start, _min_delay+1, MPI_INTEGER_T, MPI_COMM_WORLD);

	cudaDeviceSynchronize();

	ncclGroupStart();
	for (int r=0; r<_gpu_num; r++) {
		int num = _send_start[i*(_min_delay+1)+_min_delay];
		if (num > 0) {
			ncclSend(_gpu_array->_send_data + _send_offset[r_offset + r], num, NCCL_INTEGER_T, r, comm_gpu, s);
		}
		num = _recv_start[i*(_min_delay+1)+_min_delay];
		if (rc[r] > 0) {
			ncclRecv(_gpu_array->_recv_data + _recv_offset[r_offset + r], num, NCCL_INTEGER_T, r, comm_gpu, s);
		}
	}
	ncclGroupEnd();


	// print_mpi_x32(_recv_num, num_size, "Recv Num");

	for (int i=0; i<_proc_num; i++) {
		if (i/_gpu_num == _gpu_group) {
			_recv_num[i] = 0;
		} else {
			_recv_num[i] = _recv_start[i*(_min_delay+1)+_min_delay];
		}
	}

#ifdef ASYNC
	int ret = MPI_Ialltoallv(_send_data, _send_num, _send_offset , MPI_INTEGER_T, _recv_data, _recv_num, _recv_offset, MPI_INTEGER_T, MPI_COMM_WORLD, &_request);
	assert(ret == MPI_SUCCESS);
#else
	int ret = MPI_Alltoallv(_send_data, _send_num, _send_offset, MPI_INTEGER_T, _recv_data, _recv_num, _recv_offset, MPI_INTEGER_T, MPI_COMM_WORLD);
	assert(ret == MPI_SUCCESS);
#endif
}

template<typename TID, typename TSIZE>
int upload_gpu(TID *tables, TSIZE *table_sizes, TSIZE *c_table_sizes, TSIZE table_cap, int max_delay, int curr_delay, int grid, int block)
{
	if (curr_delay >= minDelay -1) {
		copyFromGPU(c_table_sizes, table_sizes, max_delay+1);

		for (int d=0; d<_min_delay; d++) {
			int delay_idx = (time-_min_delay+2+d+max_delay)%(max_delay+1);
			for (int g=0; g<_gpu_num; g++) {
				int p = _gpu_group * _gpu_num + g;
				int start = cnd->_recv_start[p*(_min_delay+1)+d_];
				int end = cnd->_recv_start[p*(_min_delay+1)+d_+1];
				if (end > start) {
					gpuMemcpy(tables + table_cap * delay_idx + c_table_sizes[delay_idx], _gpu_array->_recv_data + _recv_offset[p] + start, end - start);
					c_fired_sizes[delay_idx] += end - start;
				}
			}
		}

#ifdef ASYNC
		MPI_Status status_t;
		int ret = MPI_Wait(&request_t, &status_t);
		assert(ret == MPI_SUCCESS);
#endif

		for (int d=0; d < _min_delay; d++) {
			int delay_idx = (time-_min_delay+2+d+max_delay)%(max_delay+1);
			for (int p = 0; p<proc_num; p++) {
				int start = cnd->_recv_start[p*(_min_delay+1)+d_];
				int end = cnd->_recv_start[p*(_min_delay+1)+d_+1];
				if (end > start && (p/_gpu_num != _gpu_group)) {
					assert(c_fired_sizes[delay_idx] + end - start <= table_cap);
					copyToGPU(tables + table_cap*delay_idx + c_table_sizes[delay_idx], _recv_data + _recv_offset[p] + start, end-start);
					c_fired_sizes[delay_idx] += end - start;
				}
			}
		}
		copyToGPU(table_sizes, c_fired_sizes, max_delay+1);

		{ // Reset
			gpuMemset(_gpu_array->_recv_start, 0, _min_delay * _proc_num + _proc_num);
			gpuMemset(_gpu_array->_send_start, 0, _min_delay * _proc_num + _proc_num);

			memset_c(_recv_num, 0, node_num);
			memset_c(_send_num, 0, node_num);
		}
	}

	return 0;
}

