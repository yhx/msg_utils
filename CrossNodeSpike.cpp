
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "../utils/utils.h"
#include "msg_utils.h"
#include "CrossNodeSpike.h"

CrossNodeSpike::CrossNodeSpike()
{
	_node_num = 0;
	_min_delay = 0;

	_recv_offset = NULL;
	_recv_start = NULL;
	_recv_num = NULL;
	_recv_data = NULL;

	_send_offset = NULL;
	_send_start = NULL;
	_send_num = NULL;
	_send_data = NULL;

	gpu = NULL;

}

CrossNodeSpike::CrossNodeSpike(int node_num, int delay)
{
	assert(delay > 0);
	assert(node_num > 0);
	// printf("Delay: %d\n", delay);
	// printf("Node: %d\n", node_num);
	_node_num = node_num;
	_min_delay = delay;

	int size = delay * node_num;
	int num_p_1 = node_num + 1;

	_recv_offset = malloc_c<int>(num_p_1);
	_recv_start = malloc<int>(size+node_num);
	_recv_num = malloc_c<int>(node_num);
	_recv_data = NULL;
	
	_send_offset = malloc<int>(num_p_1);
	_send_start = malloc<int>(size+node_num);
	_send_num = malloc<int>(node_num);
	_send_data = NULL;

	reset();
}

void CrossNodeSpike::reset()
{
	int node_num = _node_num;
	int size = _min_delay * _node_num;
	memset_c(_recv_start, 0, size + node_num);
	memset_c(_recv_num, 0, node_num);

	memset_c(_send_start, 0, size + node_num);
	memset_c(_send_num, 0, node_num);
}

void CrossNodeSpike::alloc()
{
	int num = _node_num;
	int data_size = _recv_offset[num];
	// printf("Data Size1: %d\n", data_size);
	if (data_size > 0) {
		// printf("Size_t: %lu\n", sizeof(int)*data_size);
		_recv_data = malloc_c<int>(data_size);
		memset_c(_recv_data, 0, data_size);
	}

	data_size = _send_offset[num];
	// printf("Data Size2: %d\n", data_size);
	if (data_size > 0) {
		// printf("Size_t: %lu\n", sizeof(int)*data_size);
		_send_data = malloc_c<int>(data_size);
		memset(_send_data, 0, data_size);
	}
}

CrossNodeSpike::~CrossNodeSpike()
{
	free_c(_recv_offset);
	free_c(_recv_start);
	free_c(_recv_num);
	free_c(_recv_data);

	free_c(_send_offset);
	free_c(_send_start);
	free_c(_send_num);
	free_c(_send_data);

	data->_node_num = 0;
	data->_min_delay = 0;
}

int CrossNodeSpike::send(int dst, int tag, MPI_Comm comm) 
{
	int ret = 0;
	ret = MPI_Send(&(_node_num), 1, MPI_INT, dst, tag, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(&(_min_delay), 1, MPI_INT, dst, tag+1, comm);
	assert(ret == MPI_SUCCESS);

	// int size = _min_delay * _node_num;
	int num_p_1 = _node_num + 1;
	ret = MPI_Send(_recv_offset, num_p_1, MPI_INT, dst, tag+2, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(_send_offset, num_p_1, MPI_INT, dst, tag+3, comm);
	assert(ret == MPI_SUCCESS);
	return ret;
}

CrossNodeData * recvCND(int src, int tag, MPI_Comm comm)
{
	assert(0 == _node_num);
	assert(0 == _min_delay);

	int ret = 0;
	MPI_Status status;
	ret = MPI_Recv(&(_node_num), 1, MPI_INT, src, tag, comm, &status);
	assert(ret==MPI_SUCCESS);
	ret = MPI_Recv(&(_min_delay), 1, MPI_INT, src, tag+1, comm, &status);
	assert(ret==MPI_SUCCESS);

	int size = _min_delay * _node_num;
	int num_p_1 = _node_num + 1;
	_recv_offset = malloc_c<int>(num_p_1);
	ret = MPI_Recv(_recv_offset, num_p_1, MPI_INT, src, tag+2, comm, &status);
	assert(ret==MPI_SUCCESS);
	_send_offset = malloc_c<int>(num_p_1);
	ret = MPI_Recv(_send_offset, num_p_1, MPI_INT, src, tag+3, comm, &status);
	assert(ret==MPI_SUCCESS);

	_recv_start = malloc_c<int>(size + _node_num);
	_recv_num = malloc_c<int>(_node_num);

	_send_start = malloc_c<int>(size + _node_num);
	_send_num = malloc_c<int>(_node_num);

	reset();
	alloc();

	return 0;
}


int CrossNodeSpike::save(FILE *f)
{
	return 0;
}

int generateCND(Connection *conn, int *firedTable, int *firedTableSizes, int *idx2index, int *crossnode_index2idx, CrossNodeData *cnd, int node_num, int time, int gFiredTableCap, int min_delay, int delay)
{
	int delay_idx = time % (conn->maxDelay+1);
	int fired_size = firedTableSizes[delay_idx];
	for (int node=0; node<node_num; node++) {
		for (int idx=0; idx<fired_size; idx++) {
			int nid = firedTable[gFiredTableCap * delay_idx + idx];
			int tmp = idx2index[nid];
			if (tmp >= 0) {
				int map_nid = crossnode_index2idx[tmp*node_num+node];
				if (map_nid >= 0) {
					int idx_t = node * (min_delay+1) + delay + 1;
					cnd->_send_data[cnd->_send_offset[node] + cnd->_send_start[idx_t]]= map_nid;
					cnd->_send_start[idx_t]++;
				}
			}
		}
	}
	return 0;
}

#define ASYNC

int msg_cnd(CrossNodeData *cnd, MPI_Request *request)
{
	int node_num = cnd->_node_num;
	int delay = cnd->_min_delay;
	for (int i=0; i<node_num; i++) {
		cnd->_send_num[i] = cnd->_send_start[i*(delay+1)+delay];
	}

	// int num_size = delay * node_num;
	// print_mpi_x32(cnd->_send_num, num_size, "Send Num");
	// print_mpi_x32(cnd->_recv_num, num_size, "To Recv Num");

	MPI_Alltoall(cnd->_send_start, delay+1, MPI_INT, cnd->_recv_start, delay+1, MPI_INT, MPI_COMM_WORLD);

	// print_mpi_x32(cnd->_recv_num, num_size, "Recv Num");

	for (int i=0; i<node_num; i++) {
		cnd->_recv_num[i] = cnd->_recv_start[i*(delay+1)+delay];
	}

#ifdef ASYNC
	int ret = MPI_Ialltoallv(cnd->_send_data, cnd->_send_num, cnd->_send_offset , MPI_INT, cnd->_recv_data, cnd->_recv_num, cnd->_recv_offset, MPI_INT, MPI_COMM_WORLD, request);
	assert(ret == MPI_SUCCESS);
#else
	int ret = MPI_Alltoallv(cnd->_send_data, cnd->_send_num, cnd->_send_offset, MPI_INT, cnd->_recv_data, cnd->_recv_num, cnd->_recv_offset, MPI_INT, MPI_COMM_WORLD);
	assert(ret == MPI_SUCCESS);
#endif

	return ret;
}

int update_cnd(CrossNodeData *cnd, int curr_delay, MPI_Request *request) 
{
	int min_delay = cnd->_min_delay;
	if (curr_delay >= min_delay - 1) {
		msg_cnd(cnd, request);
	} else {
		for (int i=0; i<cnd->_node_num; i++) {
			cnd->_send_start[i*(min_delay+1)+curr_delay+2] = cnd->_send_start[i*(min_delay+1)+curr_delay+1];
		}
	}
	return 0;
}

int log_cnd(CrossNodeData *cnd, int time, FILE *sfile, FILE *rfile)
{
	fprintf(sfile, "%d: \n", time);
	for (int n=0; n<cnd->_node_num; n++) {
		for (int d=0; d<cnd->_min_delay; d++) {
			int start = cnd->_send_start[n*(cnd->_min_delay+1)+d];
			int end = cnd->_send_start[n*(cnd->_min_delay+1)+d+1];
			log_array_noendl(sfile, cnd->_send_data + cnd->_send_offset[n]+start, end-start);
			fprintf(sfile, "\t");
		}
		fprintf(sfile, "\n");
	}
	fprintf(sfile, "\n");
	fflush(sfile);

	fprintf(rfile, "%d: \n", time);
	for (int n=0; n<cnd->_node_num; n++) {
		for (int d=0; d<cnd->_min_delay; d++) {
			int start = cnd->_recv_start[n*(cnd->_min_delay+1)+d];
			int end = cnd->_recv_start[n*(cnd->_min_delay+1)+d+1];
			log_array_noendl(rfile, cnd->_recv_data + cnd->_recv_offset[n]+start, end-start);
			fprintf(rfile, "\t");
		}
		fprintf(rfile, "\n");
	}
	fprintf(rfile, "\n");
	fflush(rfile);
	return 0;
}
