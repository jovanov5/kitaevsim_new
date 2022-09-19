import math
import numpy as np
import scipy
from scipy import sparse
from scipy.sparse.linalg import eigsh

'''
The code denotes the 8 edges as g1, g2, etc., and similarly for vertices and plaquettes. In a different document you can find how I have labelled and oriented all the edges, vertices and plaquettes.

The six elements are encoded in 6-dimensional unit vectors as follows:

1      e      [1,0,0,0,0,0,0,0]
-1     a2     [0,1,0,0,0,0,0,0]
i      a      [0,0,1,0,0,0,0,0]
-i     a3     [0,0,0,1,0,0,0,0]
j      b      [0,0,0,0,1,0,0,0]
-j     a2b    [0,0,0,0,0,1,0,0]
k      ab     [0,0,0,0,0,0,1,0]
-k     a3b    [0,0,0,0,0,0,0,1]

'''

# These permutation matrices compute left multiplication, right multiplication and element inversion on the group elements as column vectors

# 1
e = np.identity(8)

# -1
a2 = np.array([[0,1,0,0,0,0,0,0],
               [1,0,0,0,0,0,0,0],
               [0,0,0,1,0,0,0,0],
               [0,0,1,0,0,0,0,0],
               [0,0,0,0,0,1,0,0],
               [0,0,0,0,1,0,0,0],
               [0,0,0,0,0,0,0,1],
               [0,0,0,0,0,0,1,0]])

# i
a_l = np.array([[0,0,0,1,0,0,0,0],
                [0,0,1,0,0,0,0,0],
                [1,0,0,0,0,0,0,0],
                [0,1,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,1],
                [0,0,0,0,0,0,1,0],
                [0,0,0,0,1,0,0,0],
                [0,0,0,0,0,1,0,0]])

# -i
a3_l = np.array([[0,0,1,0,0,0,0,0],
                 [0,0,0,1,0,0,0,0],
                 [0,1,0,0,0,0,0,0],
                 [1,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,1,0],
                 [0,0,0,0,0,0,0,1],
                 [0,0,0,0,0,1,0,0],
                 [0,0,0,0,1,0,0,0]])

# j
b_l = np.array([[0,0,0,0,0,1,0,0],
                [0,0,0,0,1,0,0,0],
                [0,0,0,0,0,0,1,0],
                [0,0,0,0,0,0,0,1],
                [1,0,0,0,0,0,0,0],
                [0,1,0,0,0,0,0,0],
                [0,0,0,1,0,0,0,0],
                [0,0,1,0,0,0,0,0]])

# -j
a2b_l = np.array([[0,0,0,0,1,0,0,0],
                  [0,0,0,0,0,1,0,0],
                  [0,0,0,0,0,0,0,1],
                  [0,0,0,0,0,0,1,0],
                  [0,1,0,0,0,0,0,0],
                  [1,0,0,0,0,0,0,0],
                  [0,0,1,0,0,0,0,0],
                  [0,0,0,1,0,0,0,0]])

# k
a3b_l = np.array([[0,0,0,0,0,0,0,1],
                  [0,0,0,0,0,0,1,0],
                  [0,0,0,0,0,1,0,0],
                  [0,0,0,0,1,0,0,0],
                  [0,0,1,0,0,0,0,0],
                  [0,0,0,1,0,0,0,0],
                  [1,0,0,0,0,0,0,0],
                  [0,1,0,0,0,0,0,0]])

# -k
ab_l = np.array([[0,0,0,0,0,0,1,0],
                 [0,0,0,0,0,0,0,1],
                 [0,0,0,0,1,0,0,0],
                 [0,0,0,0,0,1,0,0],
                 [0,0,0,1,0,0,0,0],
                 [0,0,1,0,0,0,0,0],
                 [0,1,0,0,0,0,0,0],
                 [1,0,0,0,0,0,0,0]])

# Right multiplication

# i
a_r = np.array([[0,0,0,1,0,0,0,0],
                [0,0,1,0,0,0,0,0],
                [1,0,0,0,0,0,0,0],
                [0,1,0,0,0,0,0,0],
                [0,0,0,0,0,0,1,0],
                [0,0,0,0,0,0,0,1],
                [0,0,0,0,0,1,0,0],
                [0,0,0,0,1,0,0,0]])

# -i
a3_r = np.array([[0,0,1,0,0,0,0,0],
                 [0,0,0,1,0,0,0,0],
                 [0,1,0,0,0,0,0,0],
                 [1,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,1],
                 [0,0,0,0,0,0,1,0],
                 [0,0,0,0,1,0,0,0],
                 [0,0,0,0,0,1,0,0]])

# j
b_r = np.array([[0,0,0,0,0,1,0,0],
                [0,0,0,0,1,0,0,0],
                [0,0,0,0,0,0,0,1],
                [0,0,0,0,0,0,1,0],
                [1,0,0,0,0,0,0,0],
                [0,1,0,0,0,0,0,0],
                [0,0,1,0,0,0,0,0],
                [0,0,0,1,0,0,0,0]])

# -j
a2b_r = np.array([[0,0,0,0,1,0,0,0],
                  [0,0,0,0,0,1,0,0],
                  [0,0,0,0,0,0,1,0],
                  [0,0,0,0,0,0,0,1],
                  [0,1,0,0,0,0,0,0],
                  [1,0,0,0,0,0,0,0],
                  [0,0,0,1,0,0,0,0],
                  [0,0,1,0,0,0,0,0]])

# k
a3b_r = np.array([[0,0,0,0,0,0,0,1],
                  [0,0,0,0,0,0,1,0],
                  [0,0,0,0,1,0,0,0],
                  [0,0,0,0,0,1,0,0],
                  [0,0,0,1,0,0,0,0],
                  [0,0,1,0,0,0,0,0],
                  [1,0,0,0,0,0,0,0],
                  [0,1,0,0,0,0,0,0]])

# -k
ab_r = np.array([[0,0,0,0,0,0,1,0],
                 [0,0,0,0,0,0,0,1],
                 [0,0,0,0,0,1,0,0],
                 [0,0,0,0,1,0,0,0],
                 [0,0,1,0,0,0,0,0],
                 [0,0,0,1,0,0,0,0],
                 [0,1,0,0,0,0,0,0],
                 [1,0,0,0,0,0,0,0]])


# This matrix acts as the inverting operator on the group elements as column vectors

inv = np.array([[1,0,0,0,0,0,0,0],
                [0,1,0,0,0,0,0,0],
                [0,0,0,1,0,0,0,0],
                [0,0,1,0,0,0,0,0],
                [0,0,0,0,0,1,0,0],
                [0,0,0,0,1,0,0,0],
                [0,0,0,0,0,0,0,1],
                [0,0,0,0,0,0,1,0]])

# This function transforms group elements from string format (as written at start of file) to column vector format used in the code. This makes some later functions more readable.

def vector(element):
    if element in ['1' , 'e']:
        return np.array([1,0,0,0,0,0,0,0])
    elif element in ['-1' , 'a2']:
        return np.array([0,1,0,0,0,0,0,0])
    elif element in ['i' , 'a']:
        return np.array([0,0,1,0,0,0,0,0])
    elif element in ['-i' , 'a3']:
        return np.array([0,0,0,1,0,0,0,0])
    elif element in ['j' , 'b']:
        return np.array([0,0,0,0,1,0,0,0])
    elif element in ['-j' , 'a2b']:
        return np.array([0,0,0,0,0,1,0,0])
    elif element in ['k' , 'ab']:
        return np.array([0,0,0,0,0,0,1,0])
    elif element in ['-k' , 'a3b']:
        return np.array([0,0,0,0,0,0,0,1])

def conjugacy_class(element):
    if element in ['1' , 'e']:
        return [vector('1')]
    elif element in ['-1' , 'a2']:
        return [vector('-1')]
    elif element in ['i' , 'a' , '-i' , 'a3']:
        return [vector('i'),vector('-i')]
    elif element in ['j' , 'b' , '-j' , 'a2b']:
        return [vector('j'),vector('-j')]
    elif element in ['k' , 'a3b' , '-k' , 'ab']:
        return [vector('k'),vector('-k')]

# Gets the correct matrices for left and right multiplication

def left_mult(element):
    if np.array_equal(element,np.array([1,0,0,0,0,0,0,0])) == True:
        return e
    elif np.array_equal(element,np.array([0,1,0,0,0,0,0,0])) == True:
        return a2
    elif np.array_equal(element,np.array([0,0,1,0,0,0,0,0])) == True:
        return a_l
    elif np.array_equal(element,np.array([0,0,0,1,0,0,0,0])) == True:
        return a3_l
    elif np.array_equal(element,np.array([0,0,0,0,1,0,0,0])) == True:
        return b_l
    elif np.array_equal(element,np.array([0,0,0,0,0,1,0,0])) == True:
        return a2b_l
    elif np.array_equal(element,np.array([0,0,0,0,0,0,1,0])) == True:
        return a3b_l
    elif np.array_equal(element,np.array([0,0,0,0,0,0,0,1])) == True:
        return ab_l

def right_mult(element):
    if np.array_equal(element,np.array([1,0,0,0,0,0,0,0])) == True:
        return e
    elif np.array_equal(element,np.array([0,1,0,0,0,0,0,0])) == True:
        return a2
    elif np.array_equal(element,np.array([0,0,1,0,0,0,0,0])) == True:
        return a_r
    elif np.array_equal(element,np.array([0,0,0,1,0,0,0,0])) == True:
        return a3_r
    elif np.array_equal(element,np.array([0,0,0,0,1,0,0,0])) == True:
        return b_r
    elif np.array_equal(element,np.array([0,0,0,0,0,1,0,0])) == True:
        return a2b_r
    elif np.array_equal(element,np.array([0,0,0,0,0,0,1,0])) == True:
        return a3b_r
    elif np.array_equal(element,np.array([0,0,0,0,0,0,0,1])) == True:
        return ab_r



# Checks for the vertex conditions

def vertex1_check(g1,g2,g3,g4):
    product = left_mult(inv @ g4) @ left_mult(g3) @ left_mult(inv @ g2) @ g1

    return np.array_equal(product, np.array([1,0,0,0,0,0,0,0]))

def vertex2_check(g1,g5,g8):
    product = left_mult(g5) @ left_mult(g8) @ (inv @ g1)

    return np.array_equal(product, np.array([1,0,0,0,0,0,0,0]))

def vertex3_check(g2,g5,g6):
    product = left_mult(inv @ g6) @ left_mult(inv @ g5) @ g2

    return np.array_equal(product, np.array([1,0,0,0,0,0,0,0]))

def vertex4_check(g3,g6,g7):
    product = left_mult(g7) @ left_mult(g6) @ (inv @ g3)

    return np.array_equal(product, np.array([1,0,0,0,0,0,0,0]))

def vertex5_check(g4,g7,g8):
    product = left_mult(inv @ g8) @ left_mult(inv @ g7) @ g4

    return np.array_equal(product, np.array([1,0,0,0,0,0,0,0]))

# Generating vertex operators as matrices, now of size 16,777,216 by 16,277,216...

def vertex1():
    vertex1 = sparse.csr_matrix((8**8,8**8))

    for i in range(8):
        g1 = np.zeros(8)
        g1[i] = 1

        for j in range(8):
            g2 = np.zeros(8)
            g2[j] = 1

            for k in range(8):
                g3 = np.zeros(8)
                g3[k] = 1

                for l in range(8):
                    g4 = np.zeros(8)
                    g4[l] = 1

                    if vertex1_check(g1,g2,g3,g4) == True:

                        projector1 = sparse.csr_matrix(np.outer(g1,g1))
                        projector2 = sparse.csr_matrix(np.outer(g2,g2))
                        projector3 = sparse.csr_matrix(np.outer(g3,g3))
                        projector4 = sparse.csr_matrix(np.outer(g4,g4))

                        projector = sparse.kron(projector1,
                                    sparse.kron(projector2,
                                    sparse.kron(projector3,
                                    sparse.kron(projector4,
                                                sparse.identity(8**4,format='csr')))))

                        vertex1 += projector


    return vertex1

def vertex2():
    vertex2 = sparse.csr_matrix((8**8,8**8))

    for i in range(8):
        g1 = np.zeros(8)
        g1[i] = 1

        for j in range(8):
            g5 = np.zeros(8)
            g5[j] = 1

            for k in range(8):
                g8 = np.zeros(8)
                g8[k] = 1


                if vertex2_check(g1,g5,g8) == True:

                    projector1 = sparse.csr_matrix(np.outer(g1,g1))
                    projector5 = sparse.csr_matrix(np.outer(g5,g5))
                    projector8 = sparse.csr_matrix(np.outer(g8,g8))

                    projector = sparse.kron(projector1,
                                sparse.kron(sparse.identity(8**3,format='csr'),
                                sparse.kron(projector5,
                                sparse.kron(sparse.identity(8**2, format='csr'),
                                            projector8))))

                    vertex2 += projector
    return vertex2

def vertex3():
    vertex3 = sparse.csr_matrix((8**8,8**8))

    for i in range(8):
        g2 = np.zeros(8)
        g2[i] = 1

        for j in range(8):
            g5 = np.zeros(8)
            g5[j] = 1

            for k in range(8):
                g6 = np.zeros(8)
                g6[k] = 1


                if vertex3_check(g2,g5,g6) == True:

                    projector2 = sparse.csr_matrix(np.outer(g2,g2))
                    projector5 = sparse.csr_matrix(np.outer(g5,g5))
                    projector6 = sparse.csr_matrix(np.outer(g6,g6))

                    projector = sparse.kron(sparse.identity(8,format='csr'),
                                sparse.kron(projector2,
                                sparse.kron(sparse.identity(8**2,format='csr'),
                                sparse.kron(projector5,
                                sparse.kron(projector6,
                                            sparse.identity(8**2,format='csr'))))))

                    vertex3 += projector

    return vertex3

def vertex4():
    vertex4 = sparse.csr_matrix((8**8,8**8))

    for i in range(8):
        g3 = np.zeros(8)
        g3[i] = 1

        for j in range(8):
            g6 = np.zeros(8)
            g6[j] = 1

            for k in range(8):
                g7 = np.zeros(8)
                g7[k] = 1


                if vertex4_check(g3,g6,g7) == True:

                    projector3 = sparse.csr_matrix(np.outer(g3,g3))
                    projector6 = sparse.csr_matrix(np.outer(g6,g6))
                    projector7 = sparse.csr_matrix(np.outer(g7,g7))

                    projector = sparse.kron(sparse.identity(8**2,format='csr'),
                                sparse.kron(projector3,
                                sparse.kron(sparse.identity(8**2,format='csr'),
                                sparse.kron(projector6,
                                sparse.kron(projector7,
                                            sparse.identity(8,format='csr'))))))

                    vertex4 += projector


    return vertex4

def vertex5():
    vertex5 = sparse.csr_matrix((8**8,8**8))

    for i in range(8):
        g4 = np.zeros(8)
        g4[i] = 1

        for j in range(8):
            g7 = np.zeros(8)
            g7[j] = 1

            for k in range(8):
                g8 = np.zeros(8)
                g8[k] = 1


                if vertex5_check(g4,g7,g8) == True:

                    projector4 = sparse.csr_matrix(np.outer(g4,g4))
                    projector7 = sparse.csr_matrix(np.outer(g7,g7))
                    projector8 = sparse.csr_matrix(np.outer(g8,g8))

                    projector = sparse.kron(sparse.identity(8**3,format='csr'),
                                sparse.kron(projector4,
                                sparse.kron(sparse.identity(8**2,format='csr'),
                                sparse.kron(projector7,
                                            projector8))))

                    vertex5 += projector

    return vertex5

# Generating plaquette operators as matrices

def plaquette1():
    plaquette1 = sparse.csr_matrix((8**8,8**8))

    for i in range(8):                                      # sum over all elements in the group
        h = np.zeros(8)
        h[i] = 1

        g1_operator = sparse.csr_matrix(left_mult(h))
        g2_operator = sparse.csr_matrix(left_mult(h))
        g5_operator = sparse.csr_matrix(left_mult(h))

        plaquette1 += sparse.kron(g1_operator,
                        sparse.kron(g2_operator,
                        sparse.kron(sparse.identity(8**2,format='csr'),
                        sparse.kron(g5_operator,
                                    sparse.identity(8**3,format='csr')))))
    plaquette1 /= 8

    return plaquette1

def plaquette2():
    plaquette2 = sparse.csr_matrix((8**8,8**8))

    for i in range(8):                                      # sum over all elements in the group
        h = np.zeros(8)
        h[i] = 1

        g2_operator = sparse.csr_matrix(right_mult(inv @ h))
        g3_operator = sparse.csr_matrix(right_mult(inv @ h))
        g6_operator = sparse.csr_matrix(right_mult(inv @ h))

        plaquette2 += sparse.kron(sparse.identity(8,format='csr'),
                        sparse.kron(g2_operator,
                        sparse.kron(g3_operator,
                        sparse.kron(sparse.identity(8**2,format='csr'),
                        sparse.kron(g6_operator,
                                    sparse.identity(8**2,format='csr'))))))
    plaquette2 /= 8

    return plaquette2

def plaquette3():
    plaquette3 = sparse.csr_matrix((8**8,8**8))

    for i in range(8):                                      # sum over all elements in the group
        h = np.zeros(8)
        h[i] = 1

        g3_operator = sparse.csr_matrix(left_mult(h))
        g4_operator = sparse.csr_matrix(left_mult(h))
        g7_operator = sparse.csr_matrix(left_mult(h))

        plaquette3 += sparse.kron(sparse.identity(8**2,format='csr'),
                        sparse.kron(g3_operator,
                        sparse.kron(g4_operator,
                        sparse.kron(sparse.identity(8**2,format='csr'),
                        sparse.kron(g7_operator,
                                    sparse.identity(8,format='csr'))))))

    plaquette3 /= 8

    return plaquette3

def plaquette4():
    plaquette4 = sparse.csr_matrix((8**8,8**8))

    for i in range(8):                                      # sum over all elements in the group
        h = np.zeros(8)
        h[i] = 1

        g1_operator = sparse.csr_matrix(right_mult(inv @ h))
        g4_operator = sparse.csr_matrix(right_mult(inv @ h))
        g8_operator = sparse.csr_matrix(right_mult(inv @ h))

        plaquette4 += sparse.kron(g1_operator,
                        sparse.kron(sparse.identity(8**2,format='csr'),
                        sparse.kron(g4_operator,
                        sparse.kron(sparse.identity(8**3,format='csr'),
                                    g8_operator))))

    plaquette4 /= 8

    return plaquette4

def outer_plaquette():

    outer_plaquette = sparse.csr_matrix((8**8,8**8))

    for i in range(8):                                      # sum over all elements in the group
        h = np.zeros(8)
        h[i] = 1

        g5_operator = sparse.csr_matrix(right_mult(inv @ h))
        g6_operator = sparse.csr_matrix(left_mult(h))
        g7_operator = sparse.csr_matrix(right_mult(inv @ h))
        g8_operator = sparse.csr_matrix(left_mult(h))

        outer_plaquette += sparse.kron(sparse.identity(8**4,format='csr'),
                            sparse.kron(g5_operator,
                            sparse.kron(g6_operator,
                            sparse.kron(g7_operator,
                                        g8_operator))))

    outer_plaquette /= 8

    return outer_plaquette

# Generating all the matrices and finding some Hamiltonian eigenvalues

vertex1 = vertex1()
vertex2 = vertex2()
vertex3 = vertex3()
vertex4 = vertex4()
vertex5 = vertex5()
plaquette1 = plaquette1()
plaquette2 = plaquette2()
plaquette3 = plaquette3()
plaquette4 = plaquette4()
outer_plaquette = outer_plaquette()


Hamiltonian = -(vertex1 + vertex2 + vertex3 + vertex4 + vertex5 + plaquette1 + plaquette2 + plaquette3 + plaquette4 + outer_plaquette)


GS = eigsh(Hamiltonian, k=1, which='SA')[1][:,0]
GS[np.abs(GS)<1e-12] = 0

# Quick checks for eigenvalues

matrices = {'vertex 1': vertex1,
            'vertex 2': vertex2,
            'vertex 3': vertex3,
            'vertex 4': vertex4,
            'vertex 5': vertex5,
            'plaquette 1': plaquette1,
            'plaquette 2': plaquette2,
            'plaquette 3': plaquette3,
            'plaquette 4': plaquette4,
            'outer plaquette': outer_plaquette,
            'Hamiltonian': Hamiltonian}

def find_EV(matrix, state):

    for i in range(-10,2):
        if np.allclose(matrix @ state, i*state, atol=1e-10,rtol=0) == True:
            return i
    return 'no eigenvalue found'

def EV_checks(state):
    for key in matrices:
        print(key, ': ', find_EV(matrices[key],state))

# perturbations

def perturbation3(element,side='l'):
    perturbation = sparse.csr_matrix((8**8,8**8))


    for h in conjugacy_class(element):

        if side == 'l':
            multiplier = sparse.csr_matrix(left_mult(h))
        elif side == 'r':
            multiplier = sparse.csr_matrix(right_mult(h))

        perturbation += sparse.kron(sparse.identity(8**2,format='csr'),
                        sparse.kron(multiplier,
                                    sparse.identity(8**5,format='csr')))

    return perturbation

def perturbation1(element,side='l'):
    perturbation = sparse.csr_matrix((8**8,8**8))


    for h in conjugacy_class(element):

        if side == 'l':
            multiplier = sparse.csr_matrix(left_mult(h))
        elif side == 'r':
            multiplier = sparse.csr_matrix(right_mult(h))

        perturbation += sparse.kron(multiplier, sparse.identity(8**7,format='csr'))

    return perturbation

def remote_perturbation24(element):

    perturbation = sparse.csr_matrix((8**8,8**8))

    for i in range(8):                              # Loop over all values g4 can take and project
        g4 = np.zeros(8)
        g4[i] = 1

        ribbon = sparse.csr_matrix((8**8,8**8))

        projector = sparse.kron(sparse.identity(8**3,format='csr'),
                    sparse.kron(sparse.csr_matrix(np.outer(g4,g4)),
                                sparse.identity(8**4,format='csr')))

        for h in conjugacy_class(element):

            multiplier1 = sparse.csr_matrix(right_mult(g4) @ right_mult(h) @ right_mult(inv @ g4))

            multiplier3 = sparse.csr_matrix(left_mult(inv @ h))

            ribbon += sparse.kron(multiplier1,
                        sparse.kron(sparse.identity(8,format='csr'),
                        sparse.kron(multiplier3,
                                    sparse.identity(8**5,format='csr'))))

        ribbon /= np.sqrt(len(conjugacy_class(element)))
        perturbation += ribbon @ projector

    return perturbation

def remote_perturbation35(element):

    perturbation = sparse.csr_matrix((8**8,8**8))

    for i in range(8):                              # Loop over all values g4 can take and project
        g1 = np.zeros(8)
        g1[i] = 1

        ribbon = sparse.csr_matrix((8**8,8**8))

        projector = sparse.kron(sparse.csr_matrix(np.outer(g1,g1)),
                                sparse.identity(8**7,format='csr'))

        for h in conjugacy_class(element):

            multiplier2 = sparse.csr_matrix(left_mult(g1) @ left_mult(inv @ h) @ left_mult(inv @ g1))

            multiplier4 = sparse.csr_matrix(right_mult(h))

            ribbon +=   sparse.kron(sparse.identity(8,format='csr'),
                        sparse.kron(multiplier2,
                        sparse.kron(sparse.identity(8,format='csr'),
                        sparse.kron(multiplier4,
                                    sparse.identity(8**4,format='csr')))))

        ribbon /= np.sqrt(len(conjugacy_class(element)))
        perturbation += ribbon @ projector

    return perturbation

def commutation():

    classes = ['i','j','k']

    for blue in classes:
        for red in classes:

            a = remote_perturbation24(blue)@remote_perturbation35(red)@GS
            b = remote_perturbation35(red)@remote_perturbation24(blue)@GS

            print('Blue: ',blue,'Red: ',red, 'Inner product: ',np.dot(a,b))


























