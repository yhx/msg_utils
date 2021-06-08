
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

	template<typename TID, typename TSIZE>
		int fetch_cpu(CrossNodeMap *map, TID *tables, TSIZE *table_sizes, TSIZE table_cap, int proc_num, int max_delay, int min_delay, int node_num, int time);
	int update_cpu(int curr_delay);
	template<typename TID, typename TSIZE>
		int upload_cpu(TID *tables, TSIZE *table_sizes, TSIZE table_cap, int max_delay, int curr_delay);

#ifdef USE_GPU
	template<typename TID, typename TSIZE>
		int fetch_gpu(CrossNodeMap *map, TID *tables, TSIZE *table_sizes, TSIZE table_cap, int proc_num, int max_delay, int min_delay, int node_num, int time, int grid, int block);
	int update_gpu(int curr_delay);
	template<typename TID, typename TSIZE>
		int upload_gpu(TID *tables, TSIZE *table_sizes, TSIZE table_cap, int max_delay, int curr_delay, int grid, int block);
#endif // USE_GPU

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
	integer_t *_recv_offset;

	// cap _proc_num + 1
	integer_t *_send_offset;

protected:
	// info
	int _proc_num;
	int _proc_rank;

	int _gpu_num;
	int _gpu_rank;
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


#endif // CROSSSPIKE_H
