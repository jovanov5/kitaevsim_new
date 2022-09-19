import math
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import eigsh

sig_X = sparse.csr_matrix(np.array([[0,1],[1, 0]]))
sig_Z = sparse.csr_matrix(np.array([[1,0],[0,-1]]))

# Vertices: pauli Z matrix acting on all 4 relevant edges

# edges 1, 3, 10, 12
vertex1 =   sparse.kron(sig_Z,
            sparse.kron(sparse.identity(2,format='csr'),
            sparse.kron(sig_Z,
            sparse.kron(sparse.identity(2**6,format='csr'),
            sparse.kron(sig_Z,
            sparse.kron(sparse.identity(2,format='csr'),
            sparse.kron(sig_Z,
                        sparse.identity(2**6,format='csr')

                        )))))))

# edges 1, 2, 13, 15
vertex2 =   sparse.kron(sig_Z,
            sparse.kron(sig_Z,
            sparse.kron(sparse.identity(2**10,format='csr'),
            sparse.kron(sig_Z,
            sparse.kron(sparse.identity(2,format='csr'),
            sparse.kron(sig_Z,
                        sparse.identity(2**3,format='csr')

                        ))))))

# edges 2, 3, 16, 18
vertex3 =   sparse.kron(sparse.identity(2,format='csr'),
            sparse.kron(sig_Z,
            sparse.kron(sig_Z,
            sparse.kron(sparse.identity(2**12,format='csr'),
            sparse.kron(sig_Z,
            sparse.kron(sparse.identity(2,format='csr'),
                        sig_Z

                        ))))))

# edges 4, 6, 10, 11
vertex4 =   sparse.kron(sparse.identity(2**3,format='csr'),
            sparse.kron(sig_Z,
            sparse.kron(sparse.identity(2,format='csr'),
            sparse.kron(sig_Z,
            sparse.kron(sparse.identity(2**3,format='csr'),
            sparse.kron(sig_Z,
            sparse.kron(sig_Z,
                        sparse.identity(2**7,format='csr')

                        )))))))

# edges 4, 5, 13, 14
vertex5 =   sparse.kron(sparse.identity(2**3,format='csr'),
            sparse.kron(sig_Z,
            sparse.kron(sig_Z,
            sparse.kron(sparse.identity(2**7,format='csr'),
            sparse.kron(sig_Z,
            sparse.kron(sig_Z,
                        sparse.identity(2**4,format='csr')

                        ))))))

# edges 5, 6, 16, 17
vertex6 =   sparse.kron(sparse.identity(2**4,format='csr'),
            sparse.kron(sig_Z,
            sparse.kron(sig_Z,
            sparse.kron(sparse.identity(2**9,format='csr'),
            sparse.kron(sig_Z,
            sparse.kron(sig_Z,
                        sparse.identity(2,format='csr')

                        ))))))

# edges 7, 9, 11, 12
vertex7 =   sparse.kron(sparse.identity(2**6,format='csr'),
            sparse.kron(sig_Z,
            sparse.kron(sparse.identity(2,format='csr'),
            sparse.kron(sig_Z,
            sparse.kron(sparse.identity(2,format='csr'),
            sparse.kron(sig_Z,
            sparse.kron(sig_Z,
                        sparse.identity(2**6,format='csr')

                        )))))))

# edges 7, 8, 14, 15
vertex8 =   sparse.kron(sparse.identity(2**6,format='csr'),
            sparse.kron(sig_Z,
            sparse.kron(sig_Z,
            sparse.kron(sparse.identity(2**5,format='csr'),
            sparse.kron(sig_Z,
            sparse.kron(sig_Z,
                        sparse.identity(2**3,format='csr')

                        ))))))

# edges 8, 9, 17, 18
vertex9 =   sparse.kron(sparse.identity(2**7,format='csr'),
            sparse.kron(sig_Z,
            sparse.kron(sig_Z,
            sparse.kron(sparse.identity(2**7,format='csr'),
            sparse.kron(sig_Z,
                        sig_Z

                        )))))

# Plaquettes: pauli X matrix acting on all 4 relevant edges

# edges 1, 4, 10, 13
plaquette1 = sparse.kron(sig_X,
             sparse.kron(sparse.identity(2**2,format='csr'),
             sparse.kron(sig_X,
             sparse.kron(sparse.identity(2**5,format='csr'),
             sparse.kron(sig_X,
             sparse.kron(sparse.identity(2**2,format='csr'),
             sparse.kron(sig_X,
                        sparse.identity(2**5,format='csr')

                        )))))))

# edges 2, 5, 13, 16

plaquette2 = sparse.kron(sparse.identity(2,format='csr'),
             sparse.kron(sig_X,
             sparse.kron(sparse.identity(2**2,format='csr'),
             sparse.kron(sig_X,
             sparse.kron(sparse.identity(2**7,format='csr'),
             sparse.kron(sig_X,
             sparse.kron(sparse.identity(2**2,format='csr'),
             sparse.kron(sig_X,
                        sparse.identity(2**2,format='csr')

                        ))))))))

# edges 3, 6, 10, 16
plaquette3 = sparse.kron(sparse.identity(2**2,format='csr'),
             sparse.kron(sig_X,
             sparse.kron(sparse.identity(2**2,format='csr'),
             sparse.kron(sig_X,
             sparse.kron(sparse.identity(2**3,format='csr'),
             sparse.kron(sig_X,
             sparse.kron(sparse.identity(2**5,format='csr'),
             sparse.kron(sig_X,
                        sparse.identity(2**2,format='csr')

                        ))))))))

# edges 4, 7, 11, 14
plaquette4 = sparse.kron(sparse.identity(2**3,format='csr'),
             sparse.kron(sig_X,
             sparse.kron(sparse.identity(2**2,format='csr'),
             sparse.kron(sig_X,
             sparse.kron(sparse.identity(2**3,format='csr'),
             sparse.kron(sig_X,
             sparse.kron(sparse.identity(2**2,format='csr'),
             sparse.kron(sig_X,
                        sparse.identity(2**4,format='csr')

                        ))))))))

# edges 5, 8, 14, 17

plaquette5 = sparse.kron(sparse.identity(2**4,format='csr'),
             sparse.kron(sig_X,
             sparse.kron(sparse.identity(2**2,format='csr'),
             sparse.kron(sig_X,
             sparse.kron(sparse.identity(2**5,format='csr'),
             sparse.kron(sig_X,
             sparse.kron(sparse.identity(2**2,format='csr'),
             sparse.kron(sig_X,
                        sparse.identity(2,format='csr')

                        ))))))))

# edges 6, 9, 11, 17
plaquette6 = sparse.kron(sparse.identity(2**5,format='csr'),
             sparse.kron(sig_X,
             sparse.kron(sparse.identity(2**2,format='csr'),
             sparse.kron(sig_X,
             sparse.kron(sparse.identity(2,format='csr'),
             sparse.kron(sig_X,
             sparse.kron(sparse.identity(2**5,format='csr'),
             sparse.kron(sig_X,
                        sparse.identity(2,format='csr')

                        ))))))))

# edges 1, 7, 12, 15
plaquette7 = sparse.kron(sig_X,
             sparse.kron(sparse.identity(2**5,format='csr'),
             sparse.kron(sig_X,
             sparse.kron(sparse.identity(2**4,format='csr'),
             sparse.kron(sig_X,
             sparse.kron(sparse.identity(2**2,format='csr'),
             sparse.kron(sig_X,
                        sparse.identity(2**3,format='csr')

                        )))))))

# edges 2, 8, 15, 18
plaquette8 = sparse.kron(sparse.identity(2,format='csr'),
             sparse.kron(sig_X,
             sparse.kron(sparse.identity(2**5,format='csr'),
             sparse.kron(sig_X,
             sparse.kron(sparse.identity(2**6,format='csr'),
             sparse.kron(sig_X,
             sparse.kron(sparse.identity(2**2,format='csr'),
                         sig_X

                        )))))))

# edges 3, 9, 12, 18
plaquette9 = sparse.kron(sparse.identity(2**2,format='csr'),
             sparse.kron(sig_X,
             sparse.kron(sparse.identity(2**5,format='csr'),
             sparse.kron(sig_X,
             sparse.kron(sparse.identity(2**2,format='csr'),
             sparse.kron(sig_X,
             sparse.kron(sparse.identity(2**5,format='csr'),
                         sig_X

                        )))))))

Hamiltonian = -(vertex1 + vertex2 + vertex3 + vertex4 + vertex5 +
                vertex6 + vertex7 + vertex8 + vertex9 +
                plaquette1 + plaquette2 + plaquette3 + plaquette4 + plaquette5 +
                plaquette6 + plaquette7 + plaquette8 + plaquette9)/2

# These functions generate the toric ground states. Because of the 4-fold ground state degeneracy, automatic diagonalisation doesn't extract these specific states, but generally results in some linear combinations. These give the 'code space' states, but sadly they take a long time to run.

GS00_init = np.kron([1,0],
            np.kron([1,0],
            np.kron([1,0],
            np.kron([1,0],
            np.kron([1,0],
            np.kron([1,0],
            np.kron([1,0],
            np.kron([1,0],
            np.kron([1,0],
            np.kron([1,0],
            np.kron([1,0],
            np.kron([1,0],
            np.kron([1,0],
            np.kron([1,0],
            np.kron([1,0],
            np.kron([1,0],
            np.kron([1,0],
                    [1,0]
             )))))))))))))))))

GS00 = np.zeros(2**18)

for i in range(512):
    GS00 += (plaquette1**int(format(i,'09b')[0]) @
             plaquette2**int(format(i,'09b')[1]) @
             plaquette3**int(format(i,'09b')[2]) @
             plaquette4**int(format(i,'09b')[3]) @
             plaquette5**int(format(i,'09b')[4]) @
             plaquette6**int(format(i,'09b')[5]) @
             plaquette7**int(format(i,'09b')[6]) @
             plaquette8**int(format(i,'09b')[7]) @
             plaquette9**int(format(i,'09b')[8])) @ GS00_init

GS00 /= 32

GS01_init = np.kron([0,1],
            np.kron([0,1],
            np.kron([0,1],
            np.kron([1,0],
            np.kron([1,0],
            np.kron([1,0],
            np.kron([1,0],
            np.kron([1,0],
            np.kron([1,0],
            np.kron([1,0],
            np.kron([1,0],
            np.kron([1,0],
            np.kron([1,0],
            np.kron([1,0],
            np.kron([1,0],
            np.kron([1,0],
            np.kron([1,0],
                    [1,0]
             )))))))))))))))))

GS01 = np.zeros(2**18)

for i in range(512):
    GS01 += (plaquette1**int(format(i,'09b')[0]) @
             plaquette2**int(format(i,'09b')[1]) @
             plaquette3**int(format(i,'09b')[2]) @
             plaquette4**int(format(i,'09b')[3]) @
             plaquette5**int(format(i,'09b')[4]) @
             plaquette6**int(format(i,'09b')[5]) @
             plaquette7**int(format(i,'09b')[6]) @
             plaquette8**int(format(i,'09b')[7]) @
             plaquette9**int(format(i,'09b')[8])) @ GS01_init

GS01 /= 32

GS10_init = np.kron([1,0],
            np.kron([1,0],
            np.kron([1,0],
            np.kron([1,0],
            np.kron([1,0],
            np.kron([1,0],
            np.kron([1,0],
            np.kron([1,0],
            np.kron([1,0],
            np.kron([0,1],
            np.kron([0,1],
            np.kron([0,1],
            np.kron([1,0],
            np.kron([1,0],
            np.kron([1,0],
            np.kron([1,0],
            np.kron([1,0],
                    [1,0]
             )))))))))))))))))

GS10 = np.zeros(2**18)

for i in range(512):
    GS10 += (plaquette1**int(format(i,'09b')[0]) @
             plaquette2**int(format(i,'09b')[1]) @
             plaquette3**int(format(i,'09b')[2]) @
             plaquette4**int(format(i,'09b')[3]) @
             plaquette5**int(format(i,'09b')[4]) @
             plaquette6**int(format(i,'09b')[5]) @
             plaquette7**int(format(i,'09b')[6]) @
             plaquette8**int(format(i,'09b')[7]) @
             plaquette9**int(format(i,'09b')[8])) @ GS10_init

GS10 /= 32

GS11_init = np.kron([0,1],
            np.kron([0,1],
            np.kron([0,1],
            np.kron([1,0],
            np.kron([1,0],
            np.kron([1,0],
            np.kron([1,0],
            np.kron([1,0],
            np.kron([1,0],
            np.kron([0,1],
            np.kron([0,1],
            np.kron([0,1],
            np.kron([1,0],
            np.kron([1,0],
            np.kron([1,0],
            np.kron([1,0],
            np.kron([1,0],
                    [1,0]
             )))))))))))))))))

GS11 = np.zeros(2**18)

for i in range(512):
    GS11 += (plaquette1**int(format(i,'09b')[0]) @
             plaquette2**int(format(i,'09b')[1]) @
             plaquette3**int(format(i,'09b')[2]) @
             plaquette4**int(format(i,'09b')[3]) @
             plaquette5**int(format(i,'09b')[4]) @
             plaquette6**int(format(i,'09b')[5]) @
             plaquette7**int(format(i,'09b')[6]) @
             plaquette8**int(format(i,'09b')[7]) @
             plaquette9**int(format(i,'09b')[8])) @ GS11_init

GS11 /= 32


matrices = {'vertex 1': vertex1,
            'vertex 2': vertex2,
            'vertex 3': vertex3,
            'vertex 4': vertex4,
            'vertex 5': vertex5,
            'vertex 6': vertex6,
            'vertex 7': vertex7,
            'vertex 8': vertex8,
            'vertex 9': vertex9,
            'plaquette 1': plaquette1,
            'plaquette 2': plaquette2,
            'plaquette 3': plaquette3,
            'plaquette 4': plaquette4,
            'plaquette 5': plaquette5,
            'plaquette 6': plaquette6,
            'plaquette 7': plaquette7,
            'plaquette 8': plaquette8,
            'plaquette 9': plaquette9,
            'Hamiltonian': Hamiltonian}

def find_EV(matrix, state):

    for i in range(-9,2):
        if np.allclose(matrix @ state, i*state, atol=1e-10,rtol=0) == True:
            return i
    return 'no eigenvalue found'

def EV_checks(state):
    for key in matrices:
        print(key, ': ', find_EV(matrices[key],state))

# number of first excited states (i.e. pairs of defects) is 45

# General function for creating defects. You give a list of edges you want to hit with either type X (pauli X for vertex defects) or type Z (pauli Z for plaquette defects).

def perturbation(edges,type):

    perturbation = sparse.identity(2**18,format='csr')
    if type == 'X':
        pauli = sig_X
    elif type == 'Z':
        pauli = sig_Z

    for edge in edges:
        if edge == 1:
            perturbation = perturbation @ sparse.kron(pauli,
                                                      sparse.identity(2**17,format='csr')
                                                      )
        elif edge == 18:
            perturbation = perturbation @ sparse.kron(sparse.identity(2**17,format='csr'),
                                                      pauli
                                                      )
        else:
            perturbation = perturbation @ sparse.kron(sparse.identity(2**(edge-1),format='csr'),
                                          sparse.kron(pauli,
                                                      sparse.identity(2**(18-edge),format='csr')
                                                      ))
    return perturbation































