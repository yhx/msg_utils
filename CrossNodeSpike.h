
#ifndef CROSSNODESPIKE_H
#define CROSSNODESPIKE_H

#include "mpi.h"

// #include "../net/Connection.h"

// Assuming node number is N, then the offset and num parameter both have N elements. offset[i] means the offset location on data array for ith node, num[i] records the actual data recived from/sended to the ith node. offset[N] is the size of data array.
class CrossNodeSpike {
public:
	CrossNodeSpike();
	CrossNodeSpike(int node_num, int delay);
	CrossNodeSpike(FILE *f);
	~CrossNodeSpike();

	int send(int dst, int tag, MPI_Comm comm);
	int recv(int src, int tag, MPI_Comm comm);
	int save(FILE *f);
	int load(FILE *f);
	int to_gpu();
	int fetch(int *tables, int *table_sizes, int *encode,  int *decode, int table_cap, int node_num, int time, int max_delay, int min_delay, int delay);
	int msg();
	int update(int curr_delay);
	int upload(int *tables, int *table_sizes, int *encode,  int *decode, int table_cap, int node_num, int time, int max_delay, int min_delay, int delay);
	int log(int time, FILE *sfile, FILE *rfile);

	void alloc();

protected:
	void reset();

public:
	// cap _node_num + 1
	int *_recv_offset;

	// cap _node_num + 1
	int *_send_offset;

protected:
	int _node_num;
	int _min_delay;

	// int _recv_size; 
	// cap _node_num * (delay+1)
	int *_recv_start;
	// cap _node_num
	int *_recv_num;
	// cap _recv_offset[_node_num]
	int *_recv_data;

	// int send_size;
	// cap _node_num * (delay+1)
	int *_send_start;
	// cap _node_num * delay
	int *_send_num;
	// cap _send_offset[_node_num]
	int *_send_data;

	CrossNodeSpike *gpu;
	MPI_Request request;

};

// void allocParaCND(CrossNodeSpike *data, int node_num, int delay);
// void allocSpikeCND(CrossNodeSpike *data);
// void resetCND(CrossNodeSpike *data);
// void freeCND(CrossNodeSpike *data);
// 
// 
// int sendCND(CrossNodeSpike *data, int dst, int tag, MPI_Comm comm);
// CrossNodeSpike * recvCND(int src, int tag, MPI_Comm comm);
// 
// int saveCND(CrossNodeSpike *data, FILE *f);
// CrossNodeSpike * loadCND(FILE *f);
// 
// CrossNodeSpike * copyCNDtoGPU(CrossNodeSpike * data);
// int freeCNDGPU(CrossNodeSpike *data);
// 
// int generateCND(Connection *conn, int *firedTable, int *firedTableSizes, int *idx2index, int *crossnode_index2idx, CrossNodeSpike *cnd, int node_num, int time, int gFiredTableCap, int min_delay, int delay);
// 
// int msg_cnd(CrossNodeSpike *cnd, MPI_Request *request);
// 
// int update_cnd(CrossNodeSpike *cnd, int curr_delay, MPI_Request *request);
// 
// int log_cnd(CrossNodeSpike *cnd, int time, FILE *sfile, FILE *rfile);
// 
// void cudaGenerateCND(Connection *conn, int *firedTable, int *firedTableSizes, int *idx2index, int *crossnode_index2idx, CrossNodeSpike *cnd, int node_num, int time, int delay, int gridSize, int blockSize);
// 
// int update_cnd_gpu(CrossNodeSpike *gpu, CrossNodeSpike *cpu, int curr_delay, MPI_Request *request);
// 
// int reset_cnd_gpu(CrossNodeSpike *gpu, CrossNodeSpike *cpu);

#endif // CROSSNODESPIKE_H
