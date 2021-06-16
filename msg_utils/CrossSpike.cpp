
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "helper/helper_c.h"
#include "helper/helper_array_c.h"
#include "msg_utils.h"
#include "CrossSpike.h"

CrossSpike::CrossSpike()
{
	_proc_num = 0;
	_proc_rank = 0;

	_gpu_num = 0;
	_gpu_rank = 0;
	_gpu_group = 0;

	_min_delay = 0;

	_recv_offset = NULL;
	_recv_start = NULL;
	_recv_num = NULL;
	_recv_data = NULL;

	_send_offset = NULL;
	_send_start = NULL;
	_send_num = NULL;
	_send_data = NULL;

	_gpu_array = NULL;
}

CrossSpike::CrossSpike(int proc_rank, int proc_num, int delay, int gpu_rank, int gpu_num, int gpu_group)
{
	assert(delay > 0);
	assert(proc_num > 0);
	assert(gpu_num > 0);
	// printf("Delay: %d\n", delay);
	// printf("Node: %d\n", proc_num);
	_proc_rank = proc_rank;
	_proc_num = proc_num;

	_gpu_rank = gpu_rank;
	_gpu_num = gpu_num;
	_gpu_group = gpu_group;

	_min_delay = delay;

	size_t size = delay * proc_num;
	size_t num_p_1 = proc_num + 1;

	_recv_offset = malloc_c<integer_t>(num_p_1);
	_recv_start = malloc_c<integer_t>(size+proc_num);
	_recv_num = malloc_c<integer_t>(proc_num);
	_recv_data = NULL;

	_send_offset = malloc_c<integer_t>(num_p_1);
	_send_start = malloc_c<integer_t>(size+proc_num);
	_send_num = malloc_c<integer_t>(proc_num);
	_send_data = NULL;

	reset();
}

void CrossSpike::reset()
{
	int proc_num = _proc_num;
	int size = _min_delay * _proc_num;
	memset_c(_recv_start, 0, size + proc_num);
	memset_c(_recv_num, 0, proc_num);

	memset_c(_send_start, 0, size + proc_num);
	memset_c(_send_num, 0, proc_num);
}

void CrossSpike::alloc()
{
	int data_size = _recv_offset[_proc_num];
	// printf("Data Size1: %d\n", data_size);
	if (data_size > 0) {
		// printf("Size_t: %lu\n", sizeof(int)*data_size);
		_recv_data = malloc_c<integer_t>(data_size);
	}

	data_size = _send_offset[_proc_num];
	// printf("Data Size2: %d\n", data_size);
	if (data_size > 0) {
		// printf("Size_t: %lu\n", sizeof(int)*data_size);
		_send_data = malloc_c<integer_t>(data_size);
	}
}


int CrossSpike::send(int dst, int tag, MPI_Comm comm) 
{
	int ret = 0;
	ret = MPI_Send(&(_proc_rank), 1, MPI_INT, dst, tag+1, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(&(_proc_num), 1, MPI_INT, dst, tag, comm);
	assert(ret == MPI_SUCCESS);

	ret = MPI_Send(&(_gpu_rank), 1, MPI_INT, dst, tag+3, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(&(_gpu_num), 1, MPI_INT, dst, tag+2, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(&(_gpu_group), 1, MPI_INT, dst, tag+4, comm);
	assert(ret == MPI_SUCCESS);

	ret = MPI_Send(&(_min_delay), 1, MPI_INT, dst, tag+5, comm);
	assert(ret == MPI_SUCCESS);

	int size = _min_delay * _proc_num;
	int num_p_1 = _proc_num + 1;

	ret = MPI_Send(_recv_offset, num_p_1, MPI_INTEGER_T, dst, tag+6, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(_recv_start, size+_proc_num, MPI_INTEGER_T, dst, tag+7, comm);
	ret = MPI_Send(_recv_num, _proc_num, MPI_INTEGER_T, dst, tag+8, comm);
	assert(ret == MPI_SUCCESS);

	ret = MPI_Send(_send_offset, num_p_1, MPI_INTEGER_T, dst, tag+9, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(_send_start, size+_proc_num, MPI_INTEGER_T, dst, tag+10, comm);
	ret = MPI_Send(_send_num, _proc_num, MPI_INTEGER_T, dst, tag+11, comm);
	assert(ret == MPI_SUCCESS);

	ret = MPI_Send(_recv_data, _recv_offset[_proc_num], MPI_INTEGER_T, dst, tag+12, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(_send_data, _send_offset[_proc_num], MPI_INTEGER_T, dst, tag+13, comm);
	assert(ret == MPI_SUCCESS);

	return 14;
}

int CrossSpike::recv(int src, int tag, MPI_Comm comm)
{
	assert(0 == _proc_num);
	assert(0 == _min_delay);

	int ret = 0;
	MPI_Status status;
	ret = MPI_Recv(&(_proc_rank), 1, MPI_INT, src, tag+1, comm, &status);
	assert(ret==MPI_SUCCESS);
	ret = MPI_Recv(&(_proc_num), 1, MPI_INT, src, tag, comm, &status);
	assert(ret==MPI_SUCCESS);

	ret = MPI_Recv(&(_gpu_rank), 1, MPI_INT, src, tag+3, comm, &status);
	assert(ret==MPI_SUCCESS);
	ret = MPI_Recv(&(_gpu_num), 1, MPI_INT, src, tag+2, comm, &status);
	assert(ret==MPI_SUCCESS);
	ret = MPI_Recv(&(_gpu_group), 1, MPI_INT, src, tag+4, comm, &status);
	assert(ret==MPI_SUCCESS);


	ret = MPI_Recv(&(_min_delay), 1, MPI_INT, src, tag+5, comm, &status);
	assert(ret==MPI_SUCCESS);

	int size = _min_delay * _proc_num;
	int num_p_1 = _proc_num + 1;

	_recv_offset = malloc_c<integer_t>(num_p_1);

	_send_offset = malloc_c<integer_t>(num_p_1);

	_recv_start = malloc_c<integer_t>(size + _proc_num);
	_recv_num = malloc_c<integer_t>(_proc_num);

	_send_start = malloc_c<integer_t>(size + _proc_num);
	_send_num = malloc_c<integer_t>(_proc_num);

	// reset();

	ret = MPI_Recv(_recv_offset, num_p_1, MPI_INTEGER_T, src, tag+6, comm, &status);
	assert(ret==MPI_SUCCESS);
	ret = MPI_Recv(_recv_start, size+_proc_num, MPI_INTEGER_T, src, tag+7, comm, &status);
	assert(ret==MPI_SUCCESS);
	ret = MPI_Recv(_recv_num, _proc_num, MPI_INTEGER_T, src, tag+8, comm, &status);
	assert(ret==MPI_SUCCESS);

	ret = MPI_Recv(_send_offset, num_p_1, MPI_INTEGER_T, src, tag+9, comm, &status);
	assert(ret==MPI_SUCCESS);
	ret = MPI_Recv(_send_start, size+_proc_num, MPI_INTEGER_T, src, tag+10, comm, &status);
	assert(ret==MPI_SUCCESS);
	ret = MPI_Recv(_send_num, _proc_num, MPI_INTEGER_T, src, tag+11, comm, &status);
	assert(ret==MPI_SUCCESS);

	alloc();

	ret = MPI_Recv(_recv_data, _recv_offset[_proc_num], MPI_INTEGER_T, src, tag+12, comm, &status);
	assert(ret==MPI_SUCCESS);
	ret = MPI_Recv(_send_data, _send_offset[_proc_num], MPI_INTEGER_T, src, tag+13, comm, &status);
	assert(ret==MPI_SUCCESS);

	return 14;
}

int CrossSpike::save(const string &path)
{
	string name = path + "/cross.data";
	FILE *f = fopen_c(name.c_str(), "w");

	fwrite_c(&(_proc_rank), 1, f);
	fwrite_c(&(_proc_num), 1, f);

	fwrite_c(&(_gpu_rank), 1, f);
	fwrite_c(&(_gpu_num), 1, f);
	fwrite_c(&(_gpu_group), 1, f);

	fwrite_c(&(_min_delay), 1, f);

	int size = _min_delay * _proc_num;
	int num_p_1 = _proc_num + 1;

	fwrite_c(_recv_offset, num_p_1, f);
	fwrite_c(_recv_start, size+_proc_num, f);
	fwrite_c(_recv_num, _proc_num, f);


	fwrite_c(_send_offset, num_p_1, f);
	fwrite_c(_send_start, size+_proc_num, f);
	fwrite_c(_send_num, _proc_num, f);

	fwrite_c(_recv_data, _recv_offset[_proc_num], f);
	fwrite_c(_send_data, _send_offset[_proc_num], f);

	fclose_c(f);

	return 0;
}

int CrossSpike::load(const string &path)
{
	string name = path + "/cross.data";
	FILE *f = fopen_c(name.c_str(), "r");

	fread_c(&(_proc_rank), 1, f);
	fread_c(&(_proc_num), 1, f);

	fread_c(&(_gpu_rank), 1, f);
	fread_c(&(_gpu_num), 1, f);
	fread_c(&(_gpu_group), 1, f);

	fread_c(&(_min_delay), 1, f);

	int size = _min_delay * _proc_num;
	int num_p_1 = _proc_num + 1;

	_recv_offset = malloc_c<integer_t>(num_p_1);
	_recv_start = malloc_c<integer_t>(size+_proc_num);
	_recv_num = malloc_c<integer_t>(_proc_num);
	_recv_data = NULL;

	_send_offset = malloc_c<integer_t>(num_p_1);
	_send_start = malloc_c<integer_t>(size+_proc_num);
	_send_num = malloc_c<integer_t>(_proc_num);
	_send_data = NULL;

	// reset();

	fread_c(_recv_offset, _proc_num+1, f);
	fread_c(_recv_start, size+_proc_num, f);
	fread_c(_recv_num, _proc_num, f);

	fread_c(_send_offset, _proc_num+1, f);
	fread_c(_send_start, size+_proc_num, f);
	fread_c(_send_num, _proc_num, f);

	alloc();
	fread_c(_recv_data, _recv_offset[_proc_num], f);
	fread_c(_send_data, _send_offset[_proc_num], f);

	fclose_c(f);

	return 0;
}

template<typename TID, typename TSIZE>
int CrossSpike::fetch(CrossMap *map, TID *tables, TSIZE *table_sizes, TSIZE table_cap, int proc_num, int max_delay, int min_delay, int node_num, int time)
{
	int delay_idx = time % (max_delay+1);
	int curr_delay = time % min_delay;
	size_t fired_size = table_sizes[delay_idx];

	for (int proc=0; proc<proc_num; proc++) {
		for (size_t idx=0; idx<fired_size; idx++) {
			fid_t nid = tables[table_cap * delay_idx + idx];
			integer_t tmp = map->idx2index[nid];
			if (tmp >= 0) {
				integer_t map_nid = map->_index2ridx[tmp*proc_num+proc];
				if (map_nid >= 0) {
					integer_t idx_t = proc * (min_delay+1) + curr_delay + 1;
					assert(idx_t >= 0);
					cnd->_send_data[cnd->_send_offset[proc] + cnd->_send_start[idx_t]]= map_nid;
					cnd->_send_start[idx_t]++;
				}
			}
		}
	}
	return 0;
}

int CrossSpike::msg_cpu()
{
	for (int i=0; i<_proc_num; i++) {
		_send_num[i] = _send_start[i*(_min_delay+1)+_min_delay];
	}

	// int num_size = _min_delay * _proc_num;
	// print_mpi_x32(_send_num, num_size, "Send Num");
	// print_mpi_x32(_recv_num, num_size, "To Recv Num");

	MPI_Alltoall(_send_start, _min_delay+1, MPI_INTEGER_T, _recv_start, _min_delay+1, MPI_INTEGER_T, MPI_COMM_WORLD);

	// print_mpi_x32(_recv_num, num_size, "Recv Num");

	for (int i=0; i<_proc_num; i++) {
		_recv_num[i] = _recv_start[i*(_min_delay+1)+_min_delay];
	}

#ifdef ASYNC
	int ret = MPI_Ialltoallv(_send_data, _send_num, _send_offset , MPI_INTEGER_T, _recv_data, _recv_num, _recv_offset, MPI_INTEGER_T, MPI_COMM_WORLD, &_request);
	assert(ret == MPI_SUCCESS);
#else
	int ret = MPI_Alltoallv(_send_data, _send_num, _send_offset, MPI_INTEGER_T, _recv_data, _recv_num, _recv_offset, MPI_INTEGER_T, MPI_COMM_WORLD);
	assert(ret == MPI_SUCCESS);
#endif

	return ret;
}

int CrossSpike::update_cpu(int curr_delay) 
{
	if (curr_delay >= _min_delay - 1) {
		msg_cpu();
	} else {
		for (int i=0; i<_proc_num; i++) {
			_send_start[i*(_min_delay+1)+curr_delay+2] = _send_start[i*(_min_delay+1)+curr_delay+1];
		}
	}
	return 0;
}

template<typename TID, typename TSIZE>
int CrossSpike::upload_cpu(TID *tables, TSIZE *table_sizes, TSIZE table_cap, int max_delay, int curr_delay);
{
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
				int end = _recv_start[p_*(_min_delay+1)+d+1];
				for (int i=start; i<end; i++) {
					fire_table[table_cap*delay_idx + table_sizes[delay_idx] + i-start] = static_cast<TID>(_recv_data[_recv_offset[p]+i]);
				}
				fired_sizes[delay_idx] += static_cast<TSIZE>(end - start);
			}
		}

		reset();
	}
}

bool CrossSpike::equal(const CrossSpike &m)
{
	bool ret = true;
	ret = ret && (_proc_rank == m._proc_rank);
	ret = ret && (_proc_size == m._proc_size);

	ret = ret && (_gpu_rank == m._gpu_rank);
	ret = ret && (_gpu_num == m._gpu_num);
	ret = ret && (_gpu_group == m._gpu_group);

	ret = ret && (_min_delay == m._min_delay);

	size_t size = _min_delay * _proc_num;
	size_t num_p_1 = _proc_num + 1;

	ret = ret && is_equal_array(_recv_offset, m._recv_offset, num_p_1);
	ret = ret && is_equal_array(_recv_start, m._recv_start, size+_proc_num);
	ret = ret && is_equal_array(_recv_num, m._recv_num, _proc_num);
	ret = ret && is_equal_array(_recv_data, m._recv_data, _recv_offset[_proc_num]);

	ret = ret && is_equal_array(_send_offset, m._send_offset, num_p_1);
	ret = ret && is_equal_array(_send_start, m._send_start, size+_proc_num);
	ret = ret && is_equal_array(_send_num, m._send_num, _proc_num);
	ret = ret && is_equal_array(_send_data, m._send_data, _send_offset[_proc_num]);

	return ret;
}

int CrossSpike::log(int time, FILE *sfile, FILE *rfile)
{
	fprintf(sfile, "%d: \n", time);
	for (int n=0; n<_proc_num; n++) {
		for (int d=0; d<_min_delay; d++) {
			int start = _send_start[n*(_min_delay+1)+d];
			int end = _send_start[n*(_min_delay+1)+d+1];
			log_array_noendl(sfile, _send_data + _send_offset[n]+start, end-start);
			fprintf(sfile, "\t");
		}
		fprintf(sfile, "\n");
	}
	fprintf(sfile, "\n");
	fflush(sfile);

	fprintf(rfile, "%d: \n", time);
	for (int n=0; n<_proc_num; n++) {
		for (int d=0; d<_min_delay; d++) {
			int start = _recv_start[n*(_min_delay+1)+d];
			int end = _recv_start[n*(_min_delay+1)+d+1];
			log_array_noendl(rfile, _recv_data + _recv_offset[n]+start, end-start);
			fprintf(rfile, "\t");
		}
		fprintf(rfile, "\n");
	}
	fprintf(rfile, "\n");
	fflush(rfile);
	return 0;
}
