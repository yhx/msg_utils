
#include "helper/helper_c.h"
#include "helper/helper_gpu.h"
#include "CrossGPUSpike.h"

CrossGPUSpike::~CrossGPUSpike()
{
	if (_node_num > 0) {
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

		_gpu_array->_node_num = 0;
		_gpu_array->_min_delay = 0;
		_gpu_array->_gpu_array = NULL;

		delete _gpu_array;
	}

	_node_num = 0;
	_min_delay = 0;
}

int CrossGPUSpike::to_gpu()
{
	size_t size = _min_delay * _node_num;
	size_t num_p_1 = _node_num + 1;

	if (!_gpu_array) {
		_gpu_array = new CrossGPUSpike;
		_gpu_array->_node_num = _node_num;
		_gpu_array->_min_delay = _min_delay;

		_gpu_array->_recv_offset = copyToGPU(_recv_offset, num_p_1);
		_gpu_array->_recv_start = copyToGPU(_recv_start, size+_node_num);
		_gpu_array->_recv_num = copyToGPU(_recv_num, _node_num);

		_gpu_array->_send_offset = copyToGPU(_send_offset, num_p_1);
		_gpu_array->_send_start = copyToGPU(_send_start, size+_node_num);
		_gpu_array->_send_num = copyToGPU(_send_num, _node_num);

		_gpu_array->_recv_data = copyToGPU(_recv_data, _recv_offset[_node_num]);

		_gpu_array->_send_data = copyToGPU(_send_data, _send_offset[_node_num]);
	} else {
		assert(_gpu_array->_node_num == _node_num);
		assert(_gpu_array->_min_delay == _min_delay);

		copyToGPU(_gpu_array->_recv_offset, _recv_offset, num_p_1);
		copyToGPU(_gpu_array->_recv_start, _recv_start, size+_node_num);
		copyToGPU(_gpu_array->_recv_num, _recv_num, _node_num);

		copyToGPU(_gpu_array->_send_offset, _send_offset, num_p_1);
		copyToGPU(_gpu_array->_send_start, _send_start, size+_node_num);
		copyToGPU(_gpu_array->_send_num, _send_num, _node_num);

		copyToGPU(_gpu_array->_recv_data, _recv_data, _recv_offset[_node_num]);

		copyToGPU(_gpu_array->_send_data, _send_data, _send_offset[_node_num]);
	}

	return 0;
}

int fetch(int *tabl

