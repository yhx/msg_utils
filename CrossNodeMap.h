
#ifndef CROSSNODEMAP_H
#define CROSSNODEMAP_H

#include "mpi.h"

#include "helper/helper_type.h"

class CrossNodeMap {
public:
	CrossNodeMap();
	CrossNodeMap(size_t num, size_t cross_size);
	CrossNodeMap(size_t num, size_t cross_num, size_t node_num);

	int send(int dest, int tag, MPI_Comm comm);
	int recv(int src, int tag, MPI_Comm comm);
	int save(FILE *f);
	int load(FILE *f);
	int compare(CrossNodeMap &m);

protected:
	// ID of neurons on this node to index in this map 
	// index = _idx2index[id]
	integer_t *_idx2index;
	// idx in this map to ID of shadow neurons on node j
	// id = _crossnode_index2idx[index * node_num + j], -1 means no shadow neuron on node j
	integer_t *_crossnodeIndex2idx;
	// _cross_size = node_num * number_of_the_neurons_on_this_node_which_have_crossnode_connections
	size_t _cross_size;
	size_t _num;
};


// int saveCNM(CrossNodeMap *map, FILE *f);
// CrossNodeMap *loadCNM(FILE *f); 
// int compareCNM(CrossNodeMap *m1, CrossNodeMap *m2);
// 
// int sendMap(CrossNodeMap * network, int dest, int tag, MPI_Comm comm);
// CrossNodeMap * recvMap(int src, int tag, MPI_Comm comm);


#endif // CROSSNODEMAP_H
