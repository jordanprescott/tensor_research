import numpy as np
from scipy.stats import unitary_group
from itertools import product
from functools import reduce

def obtain_all_prod_labels(d, N=8):
    all_prod_labels = list(product(range(d), repeat=N))
    return all_prod_labels

def ket(label, d):
    def basis(d, i):
        return np.eye(d, 1, i).flatten()
    N = len(label)
    bases = [basis(d, i) for i in label]
    ket_vec = reduce(np.kron, bases)
    return ket_vec

# down_state = np.array([1, 0])
# up_state = np.array([0, 1])

down_state = np.zeros([256, 1])
down_state[0] = 1
print(down_state)

d = len(down_state)  # dimension of local Hilbert space of each qubit (2)
interaction_dim = 2 * d

# Generate all product state labels for 8 qubits
all_prod_labels = obtain_all_prod_labels(d=2, N=8)
all_prod_kets = {}
for label in all_prod_labels:
    all_prod_kets[label] = ket(label, d)

# Generate random unitary matrices
rU1 = unitary_group.rvs(interaction_dim)
rU2 = unitary_group.rvs(interaction_dim)
rU3 = unitary_group.rvs(interaction_dim)
rU4 = unitary_group.rvs(interaction_dim)
rU5 = unitary_group.rvs(interaction_dim)
rU6 = unitary_group.rvs(interaction_dim)
rU7 = unitary_group.rvs(interaction_dim)

print(np.shape(rU1))

# Check unitarity
assert np.allclose(np.dot(rU1, rU1.conj().T), np.eye(interaction_dim))
assert np.allclose(np.dot(rU2, rU2.conj().T), np.eye(interaction_dim))
assert np.allclose(np.dot(rU3, rU3.conj().T), np.eye(interaction_dim))
assert np.allclose(np.dot(rU4, rU4.conj().T), np.eye(interaction_dim))
assert np.allclose(np.dot(rU5, rU5.conj().T), np.eye(interaction_dim))
assert np.allclose(np.dot(rU6, rU6.conj().T), np.eye(interaction_dim))
assert np.allclose(np.dot(rU7, rU7.conj().T), np.eye(interaction_dim))

# Reshape unitary matrices
unitary_shape = ([d] * 8)
rU1 = rU1.reshape(unitary_shape)
rU2 = rU2.reshape(unitary_shape)
rU3 = rU3.reshape(unitary_shape)
rU4 = rU4.reshape(unitary_shape)
rU5 = rU5.reshape(unitary_shape)
rU6 = rU6.reshape(unitary_shape)
rU7 = rU7.reshape(unitary_shape)

# Evolve the state through the network of unitaries
state_12 = np.einsum('i,j,klij', down_state, down_state, rU1)
state_34 = np.einsum('a,b,cdab', down_state, down_state, rU2)
state_56 = np.einsum('c,d,efcd', down_state, down_state, rU3)
state_78 = np.einsum('e,f,ghfe', down_state, down_state, rU4)

state_1234 = np.einsum('ij,ab,mnja', state_12, state_34, rU5)
state_5678 = np.einsum('cd,ef,pqcd', state_56, state_78, rU6)

state_final = np.einsum('ijab,cdpq,uvcd', state_1234, state_5678, rU7)

# Flatten the state vector
state_final = state_final.reshape(np.prod(np.shape(state_final)))

# Compute projections onto product states
projections_all_prod = {}
for label in all_prod_labels:
    projections_all_prod[label] = abs(np.dot(all_prod_kets[label].conj(), state_final))**2
