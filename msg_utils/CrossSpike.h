
#ifndef CROSSSPIKE_H
#define CROSSSPIKE_H

#include <string>

#include "mpi.h"
#include "nccl.h"

#include "CrossMap.h"



// #ifdef USE_GPU
#include "helper/helper_gpu.h"
#define NCCL_INTEGER_T ncclInt32
#define NCCL_UINTEGER_T ncclUint32 
// #endif

// #include "../net/Connection.h"

#define ASYNC

#ifndef INTEGER_T
typedef unsigned int uinteger_t;
typedef int integer_t;

#define INTEGER_T_MAX INT_MAX
#define UINTEGER_T_MAX UINT_MAX
#define MPI_UINTEGER_T MPI_UNSIGNED
#define MPI_INTEGER_T MPI_INT 

#endif // INTEGER_T

using std::string;

// Assuming node number is N, then the offset and num parameter both have N elements. offset[i] means the offset location on data array for ith node, num[i] records the actual data recived from/sended to the ith node. offset[N] is the size of data array.
class CrossSpike {
public:
	CrossSpike();
	CrossSpike(int proc_rank, int proc_num, int delay, int gpu_rank=0, int gpu_num=1, int gpu_group=0);
	CrossSpike(FILE *f);
	~CrossSpike();

	template<typename TID, typename TSIZE>
	int fetch_cpu(const CrossMap *map, const TID *tables, const TSIZE *table_sizes, const TSIZE &table_cap, const int &proc_num, const int &max_delay, const int &time);
	template<typename TID, typename TSIZE>
    int upload_cpu(TID *tables, TSIZE *table_sizes, const TSIZE &table_cap, const int &max_delay, const int &time);
	int update_cpu(const int &time);

// #ifdef USE_GPU
	template<typename TID, typename TSIZE>
	int fetch_gpu(const CrossMap *map, const TID *tables, const TSIZE *table_sizes, const TSIZE &table_cap, const int &proc_num, const int &max_delay, const int &time, const int &grid, const int &block);
	template<typename TID, typename TSIZE>
	int upload_gpu(TID *tables, TSIZE *table_sizes, TSIZE *c_table_sizes, const TSIZE &table_cap, const int &max_delay, const int &time, const int &grid, const int &block);
	int update_gpu(const int &curr_delay, ncclComm_t &comm_gpu, cudaStream_t &s);
// #endif // USE_GPU

	int send(int dst, int tag, MPI_Comm comm);
	int recv(int src, int tag, MPI_Comm comm);
	int save(const string &path);
	int load(const string &path);

	bool equal(const CrossSpike &m);

	int to_gpu();
	void alloc();
	int log(int time, FILE *sfile, FILE *rfile);

protected:
	int msg_cpu();
	int msg_gpu(ncclComm_t &comm_gpu, cudaStream_t &s);
	// int msg_mpi();
	void reset();

public:
	// cap _proc_num + 1
	integer_t *_recv_offset;

	// cap _proc_num + 1
	integer_t *_send_offset;

protected:
	// info
	int _proc_rank;
	int _proc_num;

	int _gpu_rank;
	int _gpu_num;
	int _gpu_group;

	integer_t _min_delay;

	// integer_t _recv_size; 
	// cap _proc_num * (delay+1)
	integer_t *_recv_start;
	// cap _proc_num
	integer_t *_recv_num;
	// cap _recv_offset[_proc_num]
	integer_t *_recv_data;

	// integer_t send_size;
	// cap _proc_num * (delay+1)
	integer_t *_send_start;
	// cap _proc_num * delay
	integer_t *_send_num;
	// cap _send_offset[_proc_num]
	integer_t *_send_data;

	MPI_Request _request;

	CrossSpike *_gpu_array;
};

template<typename TID, typename TSIZE>
int CrossSpike::fetch_cpu(const CrossMap *map, const TID *tables, const TSIZE *table_sizes, const TSIZE &table_cap, const int &proc_num, const int &max_delay, const int &time)
{
	int delay_idx = time % (max_delay+1);
	int curr_delay = time % _min_delay;
	size_t fired_size = table_sizes[delay_idx];

	for (int proc=0; proc<proc_num; proc++) {
		for (size_t idx=0; idx<fired_size; idx++) {
			TID nid = tables[table_cap * delay_idx + idx];
			integer_t tmp = map->_idx2index[nid];
			if (tmp >= 0) {
				integer_t map_nid = map->_index2ridx[tmp*proc_num+proc];
				if (map_nid >= 0) {
					integer_t idx_t = proc * (_min_delay+1) + curr_delay + 1;
					assert(idx_t >= 0);
					_send_data[_send_offset[proc] + _send_start[idx_t]]= map_nid;
					_send_start[idx_t]++;
				}
			}
		}
	}
	return 0;
}

template<typename TID, typename TSIZE>
int CrossSpike::upload_cpu(TID *tables, TSIZE *table_sizes, const TSIZE &table_cap, const int &max_delay, const int &time)
{
	int curr_delay = time % _min_delay;
	if (curr_delay >= _min_delay - 1) {
#ifdef ASYNC
		MPI_Status status_t;
		int ret = MPI_Wait(&_request, &status_t);
		assert(ret == MPI_SUCCESS);
#endif
		for (int d = 0; d < _min_delay; d++) {
			int delay_idx = (time-_min_delay+2+d+max_delay)%(max_delay+1);
			for (int p = 0; p < _proc_num; p++) {
				int start = _recv_start[p*(_min_delay+1)+d];
				int end = _recv_start[p*(_min_delay+1)+d+1];
				for (int i=start; i<end; i++) {
					tables[table_cap*delay_idx + table_sizes[delay_idx] + i-start] = static_cast<TID>(_recv_data[_recv_offset[p]+i]);
				}
				table_sizes[delay_idx] += static_cast<TSIZE>(end - start);
			}
		}

		reset();
	}
}

template<typename TID, typename TSIZE>
__global__ void fetch_kernel(integer_t *data, integer_t *offset, integer_t *num, integer_t *idx2index, integer_t *index2ridx, TID *fired_table, TSIZE *fired_sizes, TSIZE fired_cap, int proc_num, int delay_idx, int min_delay, int delay)
{
	__shared__ integer_t cross_neuron_id[MAX_BLOCK_SIZE];
	__shared__ volatile int cross_cnt;

	if (threadIdx.x == 0) {
		cross_cnt = 0;
	}
	__syncthreads();

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	TSIZE fired_size = fired_sizes[delay_idx];
	for (int node = 0; node < proc_num; node++) {
		for (int idx = tid; idx < fired_size; idx += blockDim.x * gridDim.x) {
			TID nid = static_cast<TID>(fired_table[fired_cap*delay_idx + idx]);
			integer_t tmp = idx2index[nid];
			if (tmp >= 0) {
				integer_t map_nid = index2ridx[tmp*proc_num + node];
				if (map_nid >= 0) {
					size_t test_loc = static_cast<size_t>(atomicAdd(const_cast<int*>(&cross_cnt), 1));
					if (test_loc < MAX_BLOCK_SIZE) {
						cross_neuron_id[test_loc] = static_cast<TID>(map_nid);
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
int CrossSpike::fetch_gpu(const CrossMap *map, const TID *tables, const TSIZE *table_sizes, const TSIZE &table_cap, const int &proc_num, const int &max_delay, const int &time, const int &grid, const int &block)
{
	int delay_idx = time % (max_delay + 1);
	int curr_delay = time % _min_delay;
	fetch_kernel<<<grid, block>>>(_gpu_array->_send_data, _gpu_array->_send_offset, _gpu_array->_send_start, map->_idx2index, map->_index2ridx, tables, table_sizes, table_cap, proc_num, delay_idx, _min_delay, curr_delay);
	return 0;
}

template<typename TID, typename TSIZE>
int CrossSpike::upload_gpu(TID *tables, TSIZE *table_sizes, TSIZE *c_table_sizes, const TSIZE &table_cap, const int &max_delay, const int &time, const int &grid, const int &block)
{
	int curr_delay = time % _min_delay;
	if (curr_delay >= _min_delay -1) {
		copyFromGPU(c_table_sizes, table_sizes, max_delay+1);

		for (int d=0; d<_min_delay; d++) {
			int delay_idx = (time-_min_delay+2+d+max_delay)%(max_delay+1);
			for (int g=0; g<_gpu_num; g++) {
				int p = _gpu_group * _gpu_num + g;
				int start = _recv_start[p*(_min_delay+1)+d];
				int end = _recv_start[p*(_min_delay+1)+d+1];
				if (end > start) {
					gpuMemcpy(tables + table_cap * delay_idx + c_table_sizes[delay_idx], _gpu_array->_recv_data + _recv_offset[p] + start, end - start);
					c_table_sizes[delay_idx] += end - start;
				}
			}
		}

#ifdef ASYNC
		MPI_Status status_t;
		int ret = MPI_Wait(&_request, &status_t);
		assert(ret == MPI_SUCCESS);
#endif

		for (int d=0; d < _min_delay; d++) {
			int delay_idx = (time-_min_delay+2+d+max_delay)%(max_delay+1);
			for (int p = 0; p<_proc_num; p++) {
				int start = _recv_start[p*(_min_delay+1)+d];
				int end = _recv_start[p*(_min_delay+1)+d+1];
				if (end > start && (p/_gpu_num != _gpu_group)) {
					assert(c_table_sizes[delay_idx] + end - start <= table_cap);
					copyToGPU(tables + table_cap*delay_idx + c_table_sizes[delay_idx], _recv_data + _recv_offset[p] + start, end-start);
					c_table_sizes[delay_idx] += end - start;
				}
			}
		}
		copyToGPU(table_sizes, c_table_sizes, max_delay+1);

		{ // Reset
			gpuMemset(_gpu_array->_recv_start, 0, _min_delay * _proc_num + _proc_num);
			gpuMemset(_gpu_array->_send_start, 0, _min_delay * _proc_num + _proc_num);

			memset_c(_recv_num, 0, _proc_num);
			memset_c(_send_num, 0, _proc_num);
		}
	}

	return 0;
}

#endif // CROSSSPIKE_H
