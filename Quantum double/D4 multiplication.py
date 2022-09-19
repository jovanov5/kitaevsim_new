import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

'''
The six elements are encoded in 8-dimensional unit vectors as follows:

e      [1,0,0,0,0,0,0,0]
R      [0,1,0,0,0,0,0,0]
R2     [0,0,1,0,0,0,0,0]
R3     [0,0,0,1,0,0,0,0]
m      [0,0,0,0,1,0,0,0]
mR     [0,0,0,0,0,1,0,0]
mR2    [0,0,0,0,0,0,1,0]
mR3    [0,0,0,0,0,0,0,1]

This correctly corresponds to the kronecker products of the qubit encoding in the computational basis

'''

# These permutation matrices compute left multiplication and right multiplication on the group elements as column vectors


e = np.identity(8)

R_l = np.array([[0,0,0,1,0,0,0,0],
                [1,0,0,0,0,0,0,0],
                [0,1,0,0,0,0,0,0],
                [0,0,1,0,0,0,0,0],
                [0,0,0,0,0,1,0,0],
                [0,0,0,0,0,0,1,0],
                [0,0,0,0,0,0,0,1],
                [0,0,0,0,1,0,0,0]])

R2  = np.array([[0,0,1,0,0,0,0,0],
                [0,0,0,1,0,0,0,0],
                [1,0,0,0,0,0,0,0],
                [0,1,0,0,0,0,0,0],
                [0,0,0,0,0,0,1,0],
                [0,0,0,0,0,0,0,1],
                [0,0,0,0,1,0,0,0],
                [0,0,0,0,0,1,0,0]])

R3_l = np.array([[0,1,0,0,0,0,0,0],
                 [0,0,1,0,0,0,0,0],
                 [0,0,0,1,0,0,0,0],
                 [1,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,1],
                 [0,0,0,0,1,0,0,0],
                 [0,0,0,0,0,1,0,0],
                 [0,0,0,0,0,0,1,0]])

m_l = np.array([[0,0,0,0,1,0,0,0],
                [0,0,0,0,0,1,0,0],
                [0,0,0,0,0,0,1,0],
                [0,0,0,0,0,0,0,1],
                [1,0,0,0,0,0,0,0],
                [0,1,0,0,0,0,0,0],
                [0,0,1,0,0,0,0,0],
                [0,0,0,1,0,0,0,0]])

mR_l = np.array([[0,0,0,0,0,1,0,0],
                 [0,0,0,0,0,0,1,0],
                 [0,0,0,0,0,0,0,1],
                 [0,0,0,0,1,0,0,0],
                 [0,0,0,1,0,0,0,0],
                 [1,0,0,0,0,0,0,0],
                 [0,1,0,0,0,0,0,0],
                 [0,0,1,0,0,0,0,0]])

mR2_l = np.array([[0,0,0,0,0,0,1,0],
                  [0,0,0,0,0,0,0,1],
                  [0,0,0,0,1,0,0,0],
                  [0,0,0,0,0,1,0,0],
                  [0,0,1,0,0,0,0,0],
                  [0,0,0,1,0,0,0,0],
                  [1,0,0,0,0,0,0,0],
                  [0,1,0,0,0,0,0,0]])

mR3_l = np.array([[0,0,0,0,0,0,0,1],
                  [0,0,0,0,1,0,0,0],
                  [0,0,0,0,0,1,0,0],
                  [0,0,0,0,0,0,1,0],
                  [0,1,0,0,0,0,0,0],
                  [0,0,1,0,0,0,0,0],
                  [0,0,0,1,0,0,0,0],
                  [1,0,0,0,0,0,0,0]])

R_r = np.array([[0,0,0,1,0,0,0,0],
                [1,0,0,0,0,0,0,0],
                [0,1,0,0,0,0,0,0],
                [0,0,1,0,0,0,0,0],
                [0,0,0,0,0,0,0,1],
                [0,0,0,0,1,0,0,0],
                [0,0,0,0,0,1,0,0],
                [0,0,0,0,0,0,1,0]])

R3_r = np.array([[0,1,0,0,0,0,0,0],
                 [0,0,1,0,0,0,0,0],
                 [0,0,0,1,0,0,0,0],
                 [1,0,0,0,0,0,0,0],
                 [0,0,0,0,0,1,0,0],
                 [0,0,0,0,0,0,1,0],
                 [0,0,0,0,0,0,0,1],
                 [0,0,0,0,1,0,0,0]])

m_r = np.array([[0,0,0,0,1,0,0,0],
                [0,0,0,0,0,0,0,1],
                [0,0,0,0,0,0,1,0],
                [0,0,0,0,0,1,0,0],
                [1,0,0,0,0,0,0,0],
                [0,0,0,1,0,0,0,0],
                [0,0,1,0,0,0,0,0],
                [0,1,0,0,0,0,0,0]])

mR_r = np.array([[0,0,0,0,0,1,0,0],
                 [0,0,0,0,1,0,0,0],
                 [0,0,0,0,0,0,0,1],
                 [0,0,0,0,0,0,1,0],
                 [0,1,0,0,0,0,0,0],
                 [1,0,0,0,0,0,0,0],
                 [0,0,0,1,0,0,0,0],
                 [0,0,1,0,0,0,0,0]])

mR2_r = np.array([[0,0,0,0,0,0,1,0],
                  [0,0,0,0,0,1,0,0],
                  [0,0,0,0,1,0,0,0],
                  [0,0,0,0,0,0,0,1],
                  [0,0,1,0,0,0,0,0],
                  [0,1,0,0,0,0,0,0],
                  [1,0,0,0,0,0,0,0],
                  [0,0,0,1,0,0,0,0]])

mR3_r = np.array([[0,0,0,0,0,0,0,1],
                  [0,0,0,0,0,0,1,0],
                  [0,0,0,0,0,1,0,0],
                  [0,0,0,0,1,0,0,0],
                  [0,0,0,1,0,0,0,0],
                  [0,0,1,0,0,0,0,0],
                  [0,1,0,0,0,0,0,0],
                  [1,0,0,0,0,0,0,0]])

# This function transforms group elements from string format (as written at start of file) to column vector format used in the code. 
# This makes some later functions more readable.

def vector(element):
    if element in ['e']:
        return np.array([1,0,0,0,0,0,0,0])
    elif element in ['R']:
        return np.array([0,1,0,0,0,0,0,0])
    elif element in ['R2']:
        return np.array([0,0,1,0,0,0,0,0])
    elif element in ['R3']:
        return np.array([0,0,0,1,0,0,0,0])
    elif element in ['m']:
        return np.array([0,0,0,0,1,0,0,0])
    elif element in ['mR']:
        return np.array([0,0,0,0,0,1,0,0])
    elif element in ['mR2']:
        return np.array([0,0,0,0,0,0,1,0])
    elif element in ['mR3']:
        return np.array([0,0,0,0,0,0,0,1])

def conjugacy_class(element):
    if element in ['e']:
        return [vector('e')]
    elif element in ['R2']:
        return [vector('R2')]
    elif element in ['R','R3']:
        return [vector('R'),vector('R3')]
    elif element in ['m','mR2']:
        return [vector('m'),vector('mR2')]
    elif element in ['mR','mR3']:
        return [vector('mR'),vector('mR3')]

# Gets the correct matrices for left and right multiplication

def left_mult(element):
    if np.array_equal(element,np.array([1,0,0,0,0,0,0,0])) == True:
        return e
    elif np.array_equal(element,np.array([0,1,0,0,0,0,0,0])) == True:
        return R_l
    elif np.array_equal(element,np.array([0,0,1,0,0,0,0,0])) == True:
        return R2
    elif np.array_equal(element,np.array([0,0,0,1,0,0,0,0])) == True:
        return R3_l
    elif np.array_equal(element,np.array([0,0,0,0,1,0,0,0])) == True:
        return m_l
    elif np.array_equal(element,np.array([0,0,0,0,0,1,0,0])) == True:
        return mR_l
    elif np.array_equal(element,np.array([0,0,0,0,0,0,1,0])) == True:
        return mR2_l
    elif np.array_equal(element,np.array([0,0,0,0,0,0,0,1])) == True:
        return mR3_l

def right_mult(element):
    if np.array_equal(element,np.array([1,0,0,0,0,0,0,0])) == True:
        return e
    elif np.array_equal(element,np.array([0,1,0,0,0,0,0,0])) == True:
        return R_r
    elif np.array_equal(element,np.array([0,0,1,0,0,0,0,0])) == True:
        return R2
    elif np.array_equal(element,np.array([0,0,0,1,0,0,0,0])) == True:
        return R3_r
    elif np.array_equal(element,np.array([0,0,0,0,1,0,0,0])) == True:
        return m_r
    elif np.array_equal(element,np.array([0,0,0,0,0,1,0,0])) == True:
        return mR_r
    elif np.array_equal(element,np.array([0,0,0,0,0,0,1,0])) == True:
        return mR2_r
    elif np.array_equal(element,np.array([0,0,0,0,0,0,0,1])) == True:
        return mR3_r

# Now the three-edge ground state and excited state with the conjugacy class acting on two edges.

state = np.zeros(2**9)

for i in range(8):
    g = np.zeros(8)
    g[i] = 1

    for x1 in conjugacy_class('m'):
        for x2 in conjugacy_class('m'):

            state += np.kron(right_mult(x1) @ g, np.kron(right_mult(x2) @ g, g))

state /= np.sqrt(32)

rho = np.outer(state,state)
        