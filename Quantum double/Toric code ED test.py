import math
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import eigsh

sig_X = np.array([[0, 1], [1, 0]])
sig_Z = np.array([[1,0],[0,-1]])

up = np.array([1,0])
down = np.array([0,1])

def state(edges):
    state = 1
    for element in edges:
        if element == 0:
            state = np.kron(state, [1,0])
        elif element ==1:
            state = np.kron(state, [0,1])
    return state

sample_state = state([1,0,0,1,1,0,1,1])


def edges(state):
    binary = format(np.where(state == 1)[0][0], "08b")
    edge = []
    for i in range(8):
        edge.append(int(binary[i]))
    return edge

# Vertex operators


vertex_1 = np.kron(sig_Z,               # For edges 1-4
            np.kron(sig_Z,
            np.kron(sig_Z,
            np.kron(sig_Z,
            np.identity(16)))))

vertex_2 = np.kron(sig_Z,               # For edges 1, 3, 6, 8
            np.kron(np.identity(2),
            np.kron(sig_Z,
            np.kron(np.identity(4),
            np.kron(sig_Z,
            np.kron(np.identity(2),
            sig_Z))))))

vertex_3 = np.kron(np.identity(16),     # For edges 5-8
            np.kron(sig_Z,
            np.kron(sig_Z,
            np.kron(sig_Z,
            sig_Z))))

vertex_4 = np.kron(np.identity(2),      # For edges 2, 4, 5, 7
            np.kron(sig_Z,
            np.kron(np.identity(2),
            np.kron(sig_Z,
            np.kron(sig_Z,
            np.kron(np.identity(2),
            np.kron(sig_Z,
            np.identity(2))))))))

# Plaquette operators


plaquette_1 = np.kron(sig_X,            # For edges 1, 2, 7, 8
                np.kron(sig_X,
                np.kron(np.identity(16),
                np.kron(sig_X,
                sig_X))))

plaquette_2 = np.kron(np.identity(2),   # For edges 2, 3, 5, 8
                np.kron(sig_X,
                np.kron(sig_X,
                np.kron(np.identity(2),
                np.kron(sig_X,
                np.kron(np.identity(4),
                sig_X))))))

plaquette_3 = np.kron(sig_X,            # For edges 1, 4, 6, 7
                np.kron(np.identity(4),
                np.kron(sig_X,
                np.kron(np.identity(2),
                np.kron(sig_X,
                np.kron(sig_X,
                np.identity(2)))))))

plaquette_4 = np.kron(np.identity(4),
                np.kron(sig_X,
                np.kron(sig_X,
                np.kron(sig_X,
                np.kron(sig_X,
                np.identity(4))))))

Hamiltonian = -(vertex_1 + vertex_2 + vertex_3 + vertex_4 + plaquette_1 + plaquette_2 + plaquette_3 + plaquette_4)



eigenvalues, eigenvectors = eigsh(Hamiltonian, k=10, which='SA')















