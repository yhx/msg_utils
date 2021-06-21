
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

__global__ void update_kernel(integer_t *start, int proc_num, int min_delay, int curr_delay)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i=tid; i<proc_num; i++) {
		start[i*(min_delay+1)+curr_delay+2] = start[i*(min_delay+1)+curr_delay+1];
	}
}

int CrossSpike::update_gpu(const int &curr_delay, ncclComm_t &comm_gpu, cudaStream_t &s)
{
	if (curr_delay > _min_delay -1) {
		if (_proc_num > _gpu_num) {
			copyFromGPU(_send_start, _gpu_array->_send_start, _proc_num * (_min_delay + 1));
			copyFromGPU(_send_data, _gpu_array->_send_data, _send_offset[_proc_num]);
		}
		msg_gpu(comm_gpu, s);
	} else {
		cudaDeviceSynchronize();
		update_kernel<<<1, _proc_num>>>(_gpu_array->_send_start, _proc_num, _min_delay, curr_delay);
	}

	return 0;
}

int CrossSpike::msg_gpu(ncclComm_t &comm_gpu, cudaStream_t &s)
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
		int idx = r_offset + r;
		int num = _send_start[idx*(_min_delay+1)+_min_delay];
		if (num > 0) {
			ncclSend(_gpu_array->_send_data + _send_offset[idx], num, NCCL_INTEGER_T, r, comm_gpu, s);
		}
		num = _recv_start[idx*(_min_delay+1)+_min_delay];
		if (num > 0) {
			ncclRecv(_gpu_array->_recv_data + _recv_offset[idx], num, NCCL_INTEGER_T, r, comm_gpu, s);
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

	return 0;
}


