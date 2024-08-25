import numpy as np
import quimb.tensor as qtn
import scipy
from scipy.interpolate import CubicSpline

from scipy.stats import unitary_group
from functools import reduce
from itertools import product



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
        # print(connected_nodes)
        ind_dim = tn[connected_nodes[0]].shape[tn[connected_nodes[0]].inds.index(edge_ind)]
        for i in range(ind_dim):
            stn = tn.copy()
            for cn in connected_nodes:

                indices = [slice(None) if j != stn[cn].inds.index(edge_ind) else i for j in range(len(stn[cn].inds))]
                index_tuple = tuple(indices)
                new_data = stn[cn].data[index_tuple]
                # new_data = new_data[:, np.newaxis]
                new_inds = [i for i in stn[cn].inds if i != edge_ind]

                # print(new_inds)
                # print(new_data)
                # print(np.shape(new_data))
                stn[cn].modify(inds=new_inds, data=new_data)

            sliced_tns.append(stn)
        package_list.append(SlicedPackage(edge_ind, ind_dim, connected_nodes, sliced_tns))

    return package_list


def encode(package_list, x):
    L_k = [1] + [package.ind_dim for package in package_list]
    k = 0
    deg = 0.0
    degree = []
    for sliced_package in package_list:

        e_tn = sliced_package.sliced_tns[0].copy()
        dir_indicator = 0

        for cn in sliced_package.connected_nodes:
            encoded = 0
            for j in range(sliced_package.ind_dim):
                if dir_indicator % 2 == 0:
                    encoded += sliced_package.sliced_tns[j][cn].data * x**(j * np.prod(L_k[0:k+1]))
                    # print((j * np.prod(L_k[0:k+1])))
                elif dir_indicator % 2 == 1:
                    encoded += sliced_package.sliced_tns[j][cn].data * x ** ((sliced_package.ind_dim - 1 - j) * np.prod(L_k[0:k+1]))
                    # print(((sliced_package.ind_dim - 1 - j) * np.prod(L_k[0:k+1])))

                if (j * np.prod(L_k[0:k+1])) > deg:
                    deg = (j * np.prod(L_k[0:k+1]))



            # print()
            # print(cn)
            # print(encoded)
            # print()
            dir_indicator += 1
            # all_encoded.append(tuple([cn, encoded]))
            e_tn[cn].modify(data=encoded)
            # all_encoded.append(encoded)

        degree.append(deg)
        k += 1

    # print('degree:')
    # print(2 * np.sum(degree))

    poly_degree = 2.0 * np.sum(degree)

    # encoded_tn.draw()
    # return all_encoded
    return e_tn ^ ..., poly_degree


def chebyshev_points(n):
    """
    Generate n Chebyshev points in the interval [-1, 1].

    Parameters:
    n (int): The number of Chebyshev points to generate.

    Returns:
    np.ndarray: Array of Chebyshev points.
    """
    i = np.arange(1, n+1)
    chebyshev_nodes = np.cos((2*i - 1) * np.pi / (2 * n))
    return chebyshev_nodes


#####################################


def obtain_all_prod_labels(d, N = 4):
    all_prod_labels = list(product(range(d), repeat = N))
    return all_prod_labels


def ket(label,d):
    def basis(d, i):
        return np.eye(1,d,i)
    N = len(label)
    bases = [basis(d, i) for i in label]
    ket_vec = reduce(np.kron, bases)
    return ket_vec[0]




down_state = np.array([1,0])
up_state = np.array([0,1])

d = len(down_state) #dimension of local hilbert space of each qubit (2)
interaction_dim = 2*d

all_prod_labels = obtain_all_prod_labels(d = 2, N = 4)
all_prod_kets = {}
for label in all_prod_labels:
    all_prod_kets[label] = ket(label,d)

rU1 = unitary_group.rvs(interaction_dim)
rU2 = unitary_group.rvs(interaction_dim)
rU3 = unitary_group.rvs(interaction_dim)

print(np.shape(rU1))

#check unitarity
assert np.allclose(np.dot(rU1, rU1.conj().T), np.eye(interaction_dim))
assert np.allclose(np.dot(rU2, rU2.conj().T), np.eye(interaction_dim))
assert np.allclose(np.dot(rU3, rU3.conj().T), np.eye(interaction_dim))

unitary_shape = ([d]*4)
rU1 = rU1.reshape(unitary_shape)
rU2 = rU2.reshape(unitary_shape)
rU3 = rU3.reshape(unitary_shape)


state_12 = np.einsum('i,j,klij', down_state, down_state, rU1)
state_34 = np.einsum('a,b,cdab', down_state, down_state, rU2)
state_final = np.einsum('ij,ab,mnja', state_12, state_34, rU3)

state_final = state_final.reshape(np.prod(np.shape(state_final))) #write as vector


projections_all_prod = {}
for label in all_prod_labels:
    projections_all_prod[label] = abs(np.dot(all_prod_kets[label].conj(), state_final))**2


# print(projections_all_prod)





# steps: get network, select edge, slice, encode, contract w/ diff eval points, interpolate

# get network
data = down_state
inds = ('i',)
tags = ('b0',)
b0 = qtn.Tensor(data=data, inds=inds, tags=tags)

data = down_state
inds = ('j',)
tags = ('b1',)
b1 = qtn.Tensor(data=data, inds=inds, tags=tags)

data = down_state
inds = ('a',)
tags = ('b2',)
b2 = qtn.Tensor(data=data, inds=inds, tags=tags)

data = down_state
inds = ('b',)
tags = ('b3',)
b3 = qtn.Tensor(data=data, inds=inds, tags=tags)



data = rU1
inds = ('k', 'l', 'i', 'j')
tags = ('A',)
a = qtn.Tensor(data=data, inds=inds, tags=tags)

data = rU2
inds = ('c', 'd', 'a', 'b')
tags = ('B',)
b = qtn.Tensor(data=data, inds=inds, tags=tags)

data = rU3
inds = ('m', 'n', 'l', 'd')
tags = ('C',)
c = qtn.Tensor(data=data, inds=inds, tags=tags)




TN = a & b & c & b0 & b1 & b2 & b3
# TN.draw(show_tags=True, show_inds='all')




# edges_to_slice_over = ['k', 'm']
edges_to_slice_over = ['d', 'i']
# edges_to_slice_over = ['d']



# slice non-adjacent edges
slices_packed = slice_at_ind(TN, edges_to_slice_over)
print(slices_packed)




# x = []
y = []
print("poly degree: ")

_, poly_degree = encode(slices_packed, 1)


print(poly_degree)

x = chebyshev_points(int(poly_degree)+1)
for i in range(int(poly_degree)+1):
    # print(i)
    # x.append(i)
    encoded_tn = encode(slices_packed, x[i])
    # print(encoded_tn)
    y.append(encoded_tn[0].data)
    # print(encoded_tn[0].data)

y = np.array(y)
# print(x)
# print(y)
# print(len(x))
print(np.shape(y))


print()
print("contracted value for original network: ")
print((TN ^ ...).data)
print()



# Number of data points (degree of polynomial will be n-1)
n = len(x)

# Create the Vandermonde matrix
vander_matrix = np.vander(x, n, increasing=False)

print(np.shape(vander_matrix))








# Generate the Vandermonde matrix for polynomial interpolation
vander_matrix = np.vander(x, n)

# Reshape y_points to align with the 1D structure expected for solving
# The first dimension corresponds to the number of points
dim = np.shape(y)[0]
reshaped_y = y.reshape(dim, -1)

# Solve for coefficients using np.linalg.solve
coefficients = np.linalg.solve(vander_matrix, reshaped_y)

# Reshape coefficients back to the original tensor structure
reshaped_coefficients = coefficients.reshape((n,) + y.shape[1:])







# coefficients = np.linalg.solve(vander_matrix, y)


print("predicted coeff from vandermonde interpolation:")
# print(coefficients[int(poly_degree/2)+1])
# print(coefficients[10])
print()

condition_number = np.linalg.cond(vander_matrix)
print(f"Condition number: {condition_number}")

print()
# print((poly_degree/2))
# print(coefficients[6])


# # Print the coefficients
print("Coefficients of the linalg solve interpolation:")
print(coefficients)




n_features = y.shape[1]
n_points = len(x)







# polynomial = scipy.interpolate.lagrange(x, y[:, 0])
#
# # Extract the coefficients of the polynomial
# coefficients = polynomial.coef
#
# # Print the coefficients
# print("Coefficients of the Lagrange interpolation :", coefficients)
#
# print(coefficients[10])
#
# polynomial = scipy.interpolate.lagrange(x, y[:, 1])
#
# # Extract the coefficients of the polynomial
# coefficients = polynomial.coef
#
#
# print(coefficients[10])




# state_final = (TN ^ ...).data
state_final = coefficients[4]
state_final = state_final.reshape(np.prod(np.shape(state_final))) #write as vector

# print(state_final)

projections_all_prod = {}
for label in all_prod_labels:
    projections_all_prod[label] = abs(np.dot(all_prod_kets[label].conj(), state_final))**2

all_projections_list = np.array(list(projections_all_prod.values()))
sum_all_probabilities = np.sum(all_projections_list)

print(projections_all_prod)

print('This sum should be 1:', sum_all_probabilities)
