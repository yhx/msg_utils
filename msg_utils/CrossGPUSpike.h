
#ifndef CROSSGPUSPIKE_H
#define CROSSGPUSPIKE_H

#include "mpi.h"

// #include "../net/Connection.h"

// Assuming node number is N, then the offset and num parameter both have N elements. offset[i] means the offset location on data array for ith node, num[i] records the actual data recived from/sended to the ith node. offset[N] is the size of data array.
class CrossGPUSpike {
public:
	CrossGPUSpike();
	CrossGPUSpike(int node_num, int delay);
	CrossGPUSpike(FILE *f);
	~CrossGPUSpike();

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

	CrossGPUSpike *gpu;
	MPI_Request request;

};


#endif // CROSSGPUSPIKE_H
