import math
import numpy as np
from scipy.sparse.linalg import eigsh

# The six elements are encoded in 6-dimensional vectors as e, R, R^2, m, mR, mR^2, in that order
# These matrices compute left multiplication and right multiplication

e = np.identity(6)

R2_l = np.array([[0,1,0,0,0,0],
                 [0,0,1,0,0,0],
                 [1,0,0,0,0,0],
                 [0,0,0,0,0,1],
                 [0,0,0,1,0,0],
                 [0,0,0,0,1,0]])

R_l = np.array([[0,0,1,0,0,0],
                [1,0,0,0,0,0],
                [0,1,0,0,0,0],
                [0,0,0,0,1,0],
                [0,0,0,0,0,1],
                [0,0,0,1,0,0]])

m_l = np.array([[0,0,0,1,0,0],
                [0,0,0,0,1,0],
                [0,0,0,0,0,1],
                [1,0,0,0,0,0],
                [0,1,0,0,0,0],
                [0,0,1,0,0,0]])

mR_l = np.array([[0,0,0,0,1,0],
                 [0,0,0,0,0,1],
                 [0,0,0,1,0,0],
                 [0,0,1,0,0,0],
                 [1,0,0,0,0,0],
                 [0,1,0,0,0,0]])

mR2_l = np.array([[0,0,0,0,0,1],
                  [0,0,0,1,0,0],
                  [0,0,0,0,1,0],
                  [0,1,0,0,0,0],
                  [0,0,1,0,0,0],
                  [1,0,0,0,0,0]])

R_r = np.array([[0,0,1,0,0,0],
                [1,0,0,0,0,0],
                [0,1,0,0,0,0],
                [0,0,0,0,0,1],
                [0,0,0,1,0,0],
                [0,0,0,0,1,0]])

R2_r = np.array([[0,1,0,0,0,0],
                 [0,0,1,0,0,0],
                 [1,0,0,0,0,0],
                 [0,0,0,0,1,0],
                 [0,0,0,0,0,1],
                 [0,0,0,1,0,0]])

m_r = np.array([[0,0,0,1,0,0],
                [0,0,0,0,0,1],
                [0,0,0,0,1,0],
                [1,0,0,0,0,0],
                [0,0,1,0,0,0],
                [0,1,0,0,0,0]])

mR_r = np.array([[0,0,0,0,1,0],
                 [0,0,0,1,0,0],
                 [0,0,0,0,0,1],
                 [0,1,0,0,0,0],
                 [1,0,0,0,0,0],
                 [0,0,1,0,0,0]])

mR2_r = np.array([[0,0,0,0,0,1],
                  [0,0,0,0,1,0],
                  [0,0,0,1,0,0],
                  [0,0,1,0,0,0],
                  [0,1,0,0,0,0],
                  [1,0,0,0,0,0]])

# This matrix is essentially the inverting operator

inv = np.array([[1,0,0,0,0,0],
                [0,0,1,0,0,0],
                [0,1,0,0,0,0],
                [0,0,0,1,0,0],
                [0,0,0,0,1,0],
                [0,0,0,0,0,1]])

# These functions pick out the correct matrices for left and right multiplication given a group element presented as a 6-vector.

def left_mult(element):
    if np.array_equal(element,np.array([1,0,0,0,0,0])) == True:
        return e
    elif np.array_equal(element,np.array([0,1,0,0,0,0])) == True:
        return R_l
    elif np.array_equal(element,np.array([0,0,1,0,0,0])) == True:
        return R2_l
    elif np.array_equal(element,np.array([0,0,0,1,0,0])) == True:
        return m_l
    elif np.array_equal(element,np.array([0,0,0,0,1,0])) == True:
        return mR_l
    elif np.array_equal(element,np.array([0,0,0,0,0,1])) == True:
        return mR2_l

def right_mult(element):
    if np.array_equal(element,np.array([1,0,0,0,0,0])) == True:
        return e
    elif np.array_equal(element,np.array([0,1,0,0,0,0])) == True:
        return R_r
    elif np.array_equal(element,np.array([0,0,1,0,0,0])) == True:
        return R2_r
    elif np.array_equal(element,np.array([0,0,0,1,0,0])) == True:
        return m_r
    elif np.array_equal(element,np.array([0,0,0,0,1,0])) == True:
        return mR_r
    elif np.array_equal(element,np.array([0,0,0,0,0,1])) == True:
        return mR2_r

# Preparing the vertex operators: these operators check if the vertex condition is met, i.e. if the group elements multiply to the identity, clockwise pointing outwards. This involves taking some inverses, in line with the arrow orientations I used (see book).

def vertex1_check(g1,g2,g3):
    product = left_mult(inv @ g1) @ g2
    product = left_mult(inv @ g3) @ product
    product = left_mult(g1) @ product
    return np.array_equal(product, np.array([1,0,0,0,0,0]))

def vertex2_check(g2,g3,g4):
    product = left_mult(g3) @ g4
    product = left_mult(inv @ g4) @ product
    product = left_mult(inv @ g2) @ product
    return np.array_equal(product, np.array([1,0,0,0,0,0]))

# These two functions generate the matrices that will act as vertex operators, essentially by finding all the allowed states with the functions above, generating projectors onto these states and summing up all the projectors.

def vertex1():
    vertex1 = np.zeros((1296,1296))

    # These for loops simply generate triplets of group elements in their 6-vector representation

    for i in range(6):
        g1 = np.zeros(6)
        g1[i] = 1
        for j in range(6):
            g2 = np.zeros(6)
            g2[j] = 1
            for k in range(6):
                g3 = np.zeros(6)
                g3[k] = 1

                if vertex1_check(g1,g2,g3) == True:

                # if the conditions are met, projectors are generated as outer products and patched together
                    state = np.kron(g1, np.kron(g2,g3))
                    block = np.outer(state,state)
                    projector = np.kron(block, np.identity(6))
                    vertex1 += projector
    return vertex1

# Entirely analogous procedure for the second vertex

def vertex2():
    vertex2 = np.zeros((1296,1296))

    for i in range(6):
        g2 = np.zeros(6)
        g2[i] = 1
        for j in range(6):
            g3 = np.zeros(6)
            g3[j] = 1
            for k in range(6):
                g4 = np.zeros(6)
                g4[k] = 1

                if vertex2_check(g2,g3,g4) == True:
                    state = np.kron(g2, np.kron(g3,g4))
                    block = np.outer(state,state)
                    projector = np.kron(np.identity(6),block)
                    vertex2 += projector
    return vertex2

# These two functions generate the matrices corresponding to the plaquette operators. With the arrow orientations used, and because of the simple geometry with only four edges, elements were either left-multiplied by an element h, right-multiplied by h inverse, or both (conjugated). This for all h in the group, and summed.

# The matrix for the entire system was just obtained by stitching together the matrices acting on individual edges using the tensor product.

def plaquette1():
    plaquette1 = np.zeros((1296,1296))

    for i in range(6):
        h = np.zeros(6)
        h[i] = 1

        g1_operator = right_mult(inv @ h)
        g2_operator = np.identity(6)
        g3_operator = left_mult(h) @ right_mult(inv @ h)
        g4_operator = left_mult(h)

        plaquette1 += np.kron(g1_operator,
                        np.kron(g2_operator,
                        np.kron(g3_operator,
                                g4_operator)))
    plaquette1 /= 6

    return plaquette1

# Entirely analogous procedure for the second plaquette


def plaquette2():
    plaquette2 = np.zeros((1296,1296))

    for i in range(6):
        h = np.zeros(6)
        h[i] = 1

        g1_operator = left_mult(h)
        g2_operator = left_mult(h) @ right_mult(inv @ h)
        g3_operator = np.identity(6)
        g4_operator = right_mult(inv @ h)

        plaquette2 += np.kron(g1_operator,
                        np.kron(g2_operator,
                        np.kron(g3_operator,
                                g4_operator)))
    plaquette2 /= 6

    return plaquette2

# Finally this calculates all the matrices, generates the Hamiltonian and finds the eigenvalues.

vertex1 = vertex1()
vertex2 = vertex2()
plaquette1 = plaquette1()
plaquette2 = plaquette2()

Hamiltonian = -(vertex1 + vertex2 + plaquette1 + plaquette2)

eigenvalues, eigenvectors = eigsh(Hamiltonian, k=8, which='SA')


# Checks: vertex and plaquette operators both behave like projectors in the sense that all their eigenvalues are 0 and 1 (it seems).

# Results: I find a ground state with eigenvalue -4 which is 8-fold degenerate, which corresponds to the analytical calculation in the book.

# However, there were some issues with the eigsh() method, that I will have to look into: it doesn't always accurately return only the lowest eigenvalues.



























