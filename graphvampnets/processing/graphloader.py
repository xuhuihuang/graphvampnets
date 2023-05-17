import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

class PairData(Data):
    """Time-instantaneous graph and time-lagged graph pair.

    Parameters
    ----------
    num_nodes : int
        number of nodes in each frame
    edge_index_s : ndarray
        connectivity of instant frame, shape of (num_edges_s, 2)
    edge_attr_s : ndarray
        distances of instant frame, shape of (num_edges_s, )
    edge_index_t : ndarray
        connectivity of time-lagged frame, shape of (num_edges_t, 2)
    edge_attr_t : ndarray
        distances of time-lagged frame, shape of (num_edges_t, )
    """

    def __init__(self, num_nodes, edge_attr_s, edge_index_s, edge_attr_t, edge_index_t):
        super().__init__()
        self.num_nodes = num_nodes
        self.edge_index_s = edge_index_s
        self.edge_attr_s = edge_attr_s
        self.edge_index_t = edge_index_t
        self.edge_attr_t = edge_attr_t

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_s':
            return self.num_nodes
        if key == 'edge_index_t':
            return self.num_nodes
        else:
            return super().__inc__(key, value, *args, **kwargs)
    
    def __cat_dim__(self, key, value, *args, **kwargs):
        return 0

class GraphPairLoader():
    """The dataloader used to yield batch of graph pairs.

    Parameters
    ----------
    num_atoms : int
        Number of particles in the system. 

    dataset : 

    batch_size : int
        Number of graph pairs in each batch.

    shuffle : Boolean, default = False
        Whether to shuffle the data when yielding batches.
    """

    def __init__(self, num_atoms, dataset, batch_size, shuffle=False, **kargs):
        self.num_atoms = num_atoms
        self.dataset = [
            PairData(num_atoms, torch.from_numpy(item[0][:,2]), torch.from_numpy(item[0][:,0:2]),
                     torch.from_numpy(item[1][:,2]), torch.from_numpy(item[1][:,0:2])) for item in dataset]
        self.kargs = kargs
        self.loader = DataLoader(self.dataset, batch_size, follow_batch=['edge_index_s', 'edge_index_t'], shuffle=shuffle, **self.kargs)
    
    def __iter__(self):
        for data in self.loader:
            batch_0 = torch.concat((data.edge_index_s, data.edge_attr_s.reshape(-1, 1)), dim=1)
            batch_0 = torch.concat((batch_0, torch.Tensor([-1, -1, data.num_nodes]).reshape(1, -1)))
            batch_1 = torch.concat((data.edge_index_t, data.edge_attr_t.reshape(-1, 1)), dim=1)
            batch_1 = torch.concat((batch_1, torch.Tensor([-1, -1, data.num_nodes]).reshape(1, -1)))
            yield (batch_0, batch_1)