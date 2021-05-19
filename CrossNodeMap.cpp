
#include <stdlib.h>
#include <assert.h>

#include "helper/helper_c.h"
#include "helper/helper_array_c.h"
#include "CrossNodeMap.h"

CrossNodeMap::CrossNodeMap()
{
	_idx2index = NULL;
	_crossnodeIndex2idx = NULL;
	_cross_size = 0;
	_num = 0;
}

CrossNodeMap::CrossNodeMap(size_t num, size_t cross_num, size_t node_num) : CrossNodeMap(num, cross_num * node_num)
{
}

CrossNodeMap:: CrossNodeMap(size_t num, size_t cross_size)
{
	_num = num;

	_idx2index = malloc_c<integer_t>(num);
	std::fill(_idx2index, _idx2index + num, -1);

	_cross_size = cross_size;
	if (_cross_size > 0) {
		_crossnodeIndex2idx = malloc_c<integer_t>(cross_size);
		std::fill(_crossnodeIndex2idx, _crossnodeIndex2idx + cross_size, -1);
	} else {
		_crossnodeIndex2idx = NULL;
	}
}

int CrossNodeMap::send(int dest, int tag, MPI_Comm comm) 
{
	int ret = 0;
	ret = MPI_Send(&_num, 1, MPI_SIZE_T, dest, tag, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(&_cross_size, 1, MPI_SIZE_T, dest, tag, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(_idx2index, _num, MPI_INTEGER_T, dest, tag+1, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(_crossnodeIndex2idx, _cross_size, MPI_INTEGER_T, dest, tag+2, comm);
	assert(ret == MPI_SUCCESS);
	return ret;
}

int CrossNodeMap::recv(int src, int tag, MPI_Comm comm)
{
	int ret = 0;
	MPI_Status status;
	ret = MPI_Recv(&_num, 1, MPI_SIZE_T, src, tag, comm, &status);
	assert(ret==MPI_SUCCESS);
	ret = MPI_Recv(&_cross_size, 1, MPI_SIZE_T, src, tag, comm, &status);
	assert(ret==MPI_SUCCESS);
	if (_idx2index) {
		free(_idx2index);
	}
	_idx2index = malloc_c<integer_t>(_num);
	ret = MPI_Recv(_idx2index, _num, MPI_INTEGER_T, src, tag+1, comm, &status);
	assert(ret==MPI_SUCCESS);
	_crossnodeIndex2idx = malloc_c<integer_t>(_cross_size);
	ret = MPI_Recv(_crossnodeIndex2idx, _cross_size, MPI_INTEGER_T, src, tag+2, comm, &status);
	assert(ret==MPI_SUCCESS);

	return 0;
}

int CrossNodeMap::save(FILE *f)
{
	fwrite_c(&(_num), 1, f);
	fwrite_c(&(_cross_size), 1, f);
	fwrite_c(_idx2index, _num, f);
	fwrite_c(_crossnodeIndex2idx, _cross_size, f);

	return 0;
}

int CrossNodeMap::load(FILE *f)
{
	fread_c(&(_num), 1, f);
	fread_c(&(_cross_size), 1, f);
	_idx2index = malloc_c<integer_t>(_num);
	_crossnodeIndex2idx = malloc_c<integer_t>(_cross_size);
	fwrite_c(_idx2index, _num, f);
	fwrite_c(_crossnodeIndex2idx, _cross_size, f);

	return 0;
}

int CrossNodeMap::compare(CrossNodeMap &m)
{
	bool equal = true;
	equal = (_num == m._num) && equal;
	equal = (_cross_size== m._cross_size) && equal;
	equal = is_equal_array(_idx2index, m._idx2index, _num) && equal;
	equal = is_equal_array(_crossnodeIndex2idx, m._crossnodeIndex2idx, _cross_size) && equal;
	
	return equal;
}

