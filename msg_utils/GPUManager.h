
#ifndef GPUMANAGER_H
#define GPUMANAGER_H

#include <atomic>

using std::atomic;

class GPUManager {
public:
	GPUManager();
	~GPUManager();

	int set(int id);
	int get();

	int lock();
	int unlock();

	void check();

private:
	static int _id;
	static atomic<char> _lock;
};

extern GPUManager gm;

#endif //GPUMANAGER_H
