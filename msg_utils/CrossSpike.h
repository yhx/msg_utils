
#ifndef CROSSSPIKE_H
#define CROSSSPIKE_H

#include <string>

#include "mpi.h"
#include "nccl.h"

#include "CrossMap.h"

#ifdef USE_GPU
typedef ncclInt NCCL_INTEGER_T 
typedef ncclUint32 NCCL_INTEGER_T 
#endif

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
	int upload_gpu(TID *tables, TSIZE *table_sizes, const TSIZE &table_cap, const int &max_delay, const int &curr_delay, const int &grid, const int &block);
	int update_gpu(const int &curr_delay);
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
	int msg_gpu(ncclComm_t &comm_gpu);
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


#endif // CROSSSPIKE_H
