
#include "helper/helper_c.h"
#include "helper/helper_gpu.h"
#include "CrossMap.h"

CrossMap::~CrossMap()
{
	if (_idx2index) {
		free_c(_idx2index);
	}

	if (_index2ridx) {
		free_c(_index2ridx);
	}

	if (_gpu_array) {
		if (_gpu_array->_idx2index) {
			gpuFree(_gpu_array->_idx2index);
		}

		if (_gpu_array->_index2ridx) {
			gpuFree(_gpu_array->_index2ridx);
		}

		_gpu_array->_num = 0;
		_gpu_array->_cross_size = 0;
		_gpu_array->_gpu_array = NULL;

		delete _gpu_array;
	}

	_num  = 0;
	_cross_size = 0;
}

int CrossMap::to_gpu()
{
	if (!_gpu_array) {
		_gpu_array = new CrossMap;

		_gpu_array->_num = _num;
		_gpu_array->_cross_size = _cross_size;

		_gpu_array->_idx2index = copyToGPU(_idx2index, _num);
		_gpu_array->_index2ridx = copyToGPU(_index2ridx, _cross_size);
	} else {
		assert(_gpu_array->_num == _num);
		assert(_gpu_array->_cross_size == _cross_size);

		copyToGPU(_gpu_array->_idx2index, _idx2index, _num);
		copyToGPU(_gpu_array->_index2ridx, _index2ridx, _cross_size);
	}

	return 0;
}
