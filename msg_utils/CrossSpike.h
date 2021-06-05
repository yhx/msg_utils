
#ifndef CROSSSPIKE_H
#define CROSSSPIKE_H

#include <string>

#include "mpi.h"

#include "CrossNodeMap.h"

// #include "../net/Connection.h"

#define ASYNC

using std::string;

// Assuming node number is N, then the offset and num parameter both have N elements. offset[i] means the offset location on data array for ith node, num[i] records the actual data recived from/sended to the ith node. offset[N] is the size of data array.
class CrossSpike {
public:
	CrossSpike();
	CrossSpike(int proc_num, int delay);
	CrossSpike(FILE *f);
	~CrossSpike();

	int fetch(CrossNodeMap *map, int *tables, int *table_sizes, int table_cap, int proc_num, int time, int max_delay, int min_delay, int delay);
	int update(int curr_delay);
	int upload(CrossNodeMap *map, int *tables, int *table_sizes, int table_cap, int proc_num, int time, int max_delay, int min_delay, int delay);

	int send(int dst, int tag, MPI_Comm comm);
	int recv(int src, int tag, MPI_Comm comm);
	int save(const string &path);
	int load(const string &path);
	int to_gpu();
	void alloc();
	int log(int time, FILE *sfile, FILE *rfile);

protected:
	int msg();
	int msg_gpu();
	int msg_mpi();
	void reset();

public:
	// cap _proc_num + 1
	int *_recv_offset;

	// cap _proc_num + 1
	int *_send_offset;

protected:
	// info
	int _proc_num;
	int _proc_rank;

	int _gpu_num;
	int _gpu_rank;
	int _gpu_group;

	int _min_delay;

	// int _recv_size; 
	// cap _proc_num * (delay+1)
	int *_recv_start;
	// cap _proc_num
	int *_recv_num;
	// cap _recv_offset[_proc_num]
	int *_recv_data;

	// int send_size;
	// cap _proc_num * (delay+1)
	int *_send_start;
	// cap _proc_num * delay
	int *_send_num;
	// cap _send_offset[_proc_num]
	int *_send_data;

	MPI_Request request;

	CrossSpike *_gpu_array;
};


#endif // CROSSSPIKE_H
