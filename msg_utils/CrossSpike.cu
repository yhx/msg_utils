
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

template<typename T1, typename T2, typename T3>
__global__ void fetch_kernel(T1 *data, T1 *offset, T1 *num, T2 *idx2index, T2 *index2ridx, T3 *fired_table, T3 *fired_sizes, int fired_cap, int node_num, int delay_idx, int min_delay, int delay)
{
	__shared__ T1 cross_neuron_id[MAX_BLOCK_SIZE];
	__shared__ volatile T1 cross_cnt;

	if (threadIdx.x == 0) {
		cross_cnt = 0;
	}
	__syncthreads();

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	T3 fired_size = fired_sizes[delay_idx];
	for (int node = 0; node < node_num; node++) {
		for (int idx = tid; idx < fired_size; idx += blockDim.x * gridDim.x) {
			T2 nid = fired_table[fired_cap*delay_idx + idx];
			T3 tmp = idx2index[nid];
			if (tmp >= 0) {
				T3 map_nid = index2ridx[tmp*node_num + node];
				if (map_nid >= 0) {
					size_t test_loc = static_cast<size_t>(atomicAdd((uinteger_t *)&cross_cnt, 1));
					if (test_loc < MAX_BLOCK_SIZE) {
						cross_neuron_id[test_loc] = static_cast<T1>(map_nid);
					}
				}
			}
			__syncthreads();

			if (cross_cnt > 0) {
				int idx_t = node * (min_delay + 1) + delay + 1;
				merge2array(cross_neuron_id, cross_cnt, data + offset[node] + num[idx_t], &(num[idx_t]), static_cast<T1>(fired_cap*node));
				if (threadIdx.x == 0) {
					cross_cnt = 0;
				}
			}
			__syncthreads();
		}
		__syncthreads();
	}
}

int CrossSpike::fetch(CrossNodeMap *map, uinteger_t *tables, uinteger_t *table_sizes, int table_cap, int node_num, int max_delay, int delay, int time, int grid, int block)
{
	int delay_idx = time % (max_delay + 1);
	fetch_kernel<<<grid, block, 0, _s>>>(_gpu_array->_send_data, _gpu_array->send_offset, _gpu_array->_send_start, map->_idx2index, map->_crossnodeIndex2idx, tables, table_sizes, table_cap, node_num, delay_idx, _min_delay, delay)
}

__global__ void update_kernel(int *start, int node_num, int min_delay, int curr_delay)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i=tid; i<node_num; i++) {
		start[i*(min_delay+1)+curr_delay+2] = start[i*(min_delay+1)+curr_delay+1];
	}
}

int CrossSpike::update_gpu(int curr_delay)
{
	if (curr_delay > min_delay -1) {
		msg();
	} else {
		cudaStreamSynchronize(s);
		update_kernel<<<1, _proc_num, 0, _s>>>(_gpu_array->_send_start, _proc_num, _min_delay, curr_delay);
	}

	return 0;
}

int CrossSpike::msg_gpu(ncclComm_t &comm_gpu)
{
	cudaStreamSynchronize(s);
	ncclGroupStart();
	int size = _min_delay + 1;
	for (int r=0; r<_proc_num; r++) {
		ncclSend(_gpu_array->_send_start+(r*size), size, ncclInt, r, comm_gpu, s);
		ncclRecv(_gpu_array->_recv_start+(r*size), size, ncclInt, r, comm_gpu, s);
	}
	ncclGroupEnd();
	// cudaStreamSynchronize(s);

	ncclGroupStart();
	for (int r=0; r<size_gpu; r++) {
		if (sc[r] > 0) {
			ncclSend(sb_gpu, sc[r], ncclFloat, r, comm_gpu, s);
		}
		if (rc[r] > 0) {
			ncclRecv(rb_gpu, rc[r], ncclFloat, r, comm_gpu, s);
		}
	}
	ncclGroupEnd();
	cudaStreamSynchronize(s);
}
