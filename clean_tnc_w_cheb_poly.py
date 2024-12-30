import numpy as np
import quimb.tensor as qtn
from numpy.polynomial.chebyshev import Chebyshev

class SlicedPackage:
    def __init__(self, edge_ind, ind_dim, connected_nodes, sliced_tns):
        self.edge_ind = edge_ind
        self.ind_dim = ind_dim
        self.connected_nodes = connected_nodes
        self.sliced_tns = sliced_tns

def slice_at_ind(tn, edge_inds):
    package_list = []
    for edge_ind in edge_inds:
        sliced_tns = []
        connected_nodes = [node.tags for node in tn if edge_ind in node.inds]
        ind_dim = tn[connected_nodes[0]].shape[tn[connected_nodes[0]].inds.index(edge_ind)]
        for i in range(ind_dim):
            stn = tn.copy()
            for cn in connected_nodes:
                indices = [slice(None) if j != stn[cn].inds.index(edge_ind) else i for j in range(len(stn[cn].inds))]
                index_tuple = tuple(indices)
                new_data = stn[cn].data[index_tuple]
                new_inds = [ind for ind in stn[cn].inds if ind != edge_ind]
                stn[cn].modify(inds=new_inds, data=new_data)
            sliced_tns.append(stn)
        package_list.append(SlicedPackage(edge_ind, ind_dim, connected_nodes, sliced_tns))
    return package_list

def encode(package_list, x):
    for sliced_package in package_list:
        e_tn = sliced_package.sliced_tns[0].copy()
        for cn in sliced_package.connected_nodes:
            encoded = 0
            for i, stn in enumerate(sliced_package.sliced_tns):
                data_shape = stn[cn].data.shape
                encoded += np.array([c * Chebyshev.basis(i + 1)(x) for c in stn[cn].data.ravel()], dtype=object).reshape(data_shape)
            e_tn[cn].modify(data=encoded)
    return e_tn ^ ...

# Tensor network creation
data = np.random.rand(2, 15)
a = qtn.Tensor(data=data, inds=('i', 'j'), tags=('a',))
data = np.random.rand(15, 3)
b = qtn.Tensor(data=data, inds=('j', 'k'), tags=('b',))
data = np.random.rand(3, 12)
c = qtn.Tensor(data=data, inds=('k', 'l'), tags=('c',))
data = np.random.rand(12, 50)
d = qtn.Tensor(data=data, inds=('l', 'm'), tags=('d',))
data = np.random.rand(50, 2)
e = qtn.Tensor(data=data, inds=('m', 'n'), tags=('e',))

TN = a & b & c & d & e

# Slice edges
to_slice = ['j', 'l']
slices_packed = slice_at_ind(TN, to_slice)

# Gauss-Chebyshev quadrature points and weights
n_points = 150
x_values = np.cos((2 * np.arange(1, n_points + 1) - 1) * np.pi / (2 * n_points))
weights = 2 / n_points

# Original contracted value
original_value = (TN ^ ...).data
print("Original contracted value:")
print(original_value)

# Encode and compute weighted sum of contractions
y = [weights * encode(slices_packed, x).data for x in x_values]
summed_values = sum(y)

print("Sum of contracted values:")
print(summed_values)

# Compute error
error = np.abs(summed_values - original_value)
print("Error:")
print(error)
print("Sum of error:")
print(np.sum(error))
