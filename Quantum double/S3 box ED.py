import numpy as np
from scipy import sparse
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

# This function transforms group elements from string format (as written as start of file) to column vector format used in the code.

vector = {'e':  np.array([1,0,0,0,0,0]),
          'R':   np.array([0,1,0,0,0,0]),
          'R2':  np.array([0,0,1,0,0,0]),
          'm':   np.array([0,0,0,1,0,0]),
          'mR':  np.array([0,0,0,0,1,0]),
          'mR2': np.array([0,0,0,0,0,1])

          }

conjugacy_class = {'e':   [ vector['e'] ],
                   'R':   [ vector['R'], vector['R2'] ],
                   'R2':  [ vector['R'], vector['R2'] ],
                   'm':   [ vector['m'], vector['mR'], vector['mR2'] ],
                   'mR':  [ vector['m'], vector['mR'], vector['mR2'] ],
                   'mR2': [ vector['m'], vector['mR'], vector['mR2'] ]

                    }

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

# Checks for the vertex conditions

def vertex1_check(g1,g2,g3,g4):
    product = left_mult(inv @ g4) @ left_mult(g3) @ left_mult(inv @ g2) @ g1

    return np.array_equal(product, np.array([1,0,0,0,0,0]))

def vertex2_check(g1,g5,g8):
    product = left_mult(g5) @ left_mult(g8) @ (inv @ g1)

    return np.array_equal(product, np.array([1,0,0,0,0,0]))

def vertex3_check(g2,g5,g6):
    product = left_mult(inv @ g6) @ left_mult(inv @ g5) @ g2

    return np.array_equal(product, np.array([1,0,0,0,0,0]))

def vertex4_check(g3,g6,g7):
    product = left_mult(g7) @ left_mult(g6) @ (inv @ g3)

    return np.array_equal(product, np.array([1,0,0,0,0,0]))

def vertex5_check(g4,g7,g8):
    product = left_mult(inv @ g8) @ left_mult(inv @ g7) @ g4

    return np.array_equal(product, np.array([1,0,0,0,0,0]))

# Generating vertex operators as matrices, now of size 1,679,616 by 1,679,616...

def vertex1():
    vertex1 = sparse.csr_matrix((6**8,6**8))

    for i in range(6):
        g1 = np.zeros(6)
        g1[i] = 1

        for j in range(6):
            g2 = np.zeros(6)
            g2[j] = 1

            for k in range(6):
                g3 = np.zeros(6)
                g3[k] = 1

                for l in range(6):
                    g4 = np.zeros(6)
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
                                                sparse.identity(6**4,format='csr')))))

                        vertex1 += projector


    return vertex1

def vertex2():
    vertex2 = sparse.csr_matrix((6**8,6**8))

    for i in range(6):
        g1 = np.zeros(6)
        g1[i] = 1

        for j in range(6):
            g5 = np.zeros(6)
            g5[j] = 1

            for k in range(6):
                g8 = np.zeros(6)
                g8[k] = 1


                if vertex2_check(g1,g5,g8) == True:

                    projector1 = sparse.csr_matrix(np.outer(g1,g1))
                    projector5 = sparse.csr_matrix(np.outer(g5,g5))
                    projector8 = sparse.csr_matrix(np.outer(g8,g8))

                    projector = sparse.kron(projector1,
                                sparse.kron(sparse.identity(6**3,format='csr'),
                                sparse.kron(projector5,
                                sparse.kron(sparse.identity(6**2, format='csr'),
                                            projector8))))

                    vertex2 += projector
    return vertex2

def vertex3():
    vertex3 = sparse.csr_matrix((6**8,6**8))

    for i in range(6):
        g2 = np.zeros(6)
        g2[i] = 1

        for j in range(6):
            g5 = np.zeros(6)
            g5[j] = 1

            for k in range(6):
                g6 = np.zeros(6)
                g6[k] = 1


                if vertex3_check(g2,g5,g6) == True:

                    projector2 = sparse.csr_matrix(np.outer(g2,g2))
                    projector5 = sparse.csr_matrix(np.outer(g5,g5))
                    projector6 = sparse.csr_matrix(np.outer(g6,g6))

                    projector = sparse.kron(sparse.identity(6,format='csr'),
                                sparse.kron(projector2,
                                sparse.kron(sparse.identity(6**2,format='csr'),
                                sparse.kron(projector5,
                                sparse.kron(projector6,
                                            sparse.identity(6**2,format='csr'))))))

                    vertex3 += projector

    return vertex3

def vertex4():
    vertex4 = sparse.csr_matrix((6**8,6**8))

    for i in range(6):
        g3 = np.zeros(6)
        g3[i] = 1

        for j in range(6):
            g6 = np.zeros(6)
            g6[j] = 1

            for k in range(6):
                g7 = np.zeros(6)
                g7[k] = 1


                if vertex4_check(g3,g6,g7) == True:

                    projector3 = sparse.csr_matrix(np.outer(g3,g3))
                    projector6 = sparse.csr_matrix(np.outer(g6,g6))
                    projector7 = sparse.csr_matrix(np.outer(g7,g7))

                    projector = sparse.kron(sparse.identity(6**2,format='csr'),
                                sparse.kron(projector3,
                                sparse.kron(sparse.identity(6**2,format='csr'),
                                sparse.kron(projector6,
                                sparse.kron(projector7,
                                            sparse.identity(6,format='csr'))))))

                    vertex4 += projector


    return vertex4

def vertex5():
    vertex5 = sparse.csr_matrix((6**8,6**8))

    for i in range(6):
        g4 = np.zeros(6)
        g4[i] = 1

        for j in range(6):
            g7 = np.zeros(6)
            g7[j] = 1

            for k in range(6):
                g8 = np.zeros(6)
                g8[k] = 1


                if vertex5_check(g4,g7,g8) == True:

                    projector4 = sparse.csr_matrix(np.outer(g4,g4))
                    projector7 = sparse.csr_matrix(np.outer(g7,g7))
                    projector8 = sparse.csr_matrix(np.outer(g8,g8))

                    projector = sparse.kron(sparse.identity(6**3,format='csr'),
                                sparse.kron(projector4,
                                sparse.kron(sparse.identity(6**2,format='csr'),
                                sparse.kron(projector7,
                                            projector8))))

                    vertex5 += projector

    return vertex5

# Generating plaquette operators as matrices

def plaquette1():
    plaquette1 = sparse.csr_matrix((6**8,6**8))

    for i in range(6):                                      # sum over all elements in the group
        h = np.zeros(6)
        h[i] = 1

        g1_operator = sparse.csr_matrix(left_mult(h))
        g2_operator = sparse.csr_matrix(left_mult(h))
        g5_operator = sparse.csr_matrix(left_mult(h))

        plaquette1 += sparse.kron(g1_operator,
                        sparse.kron(g2_operator,
                        sparse.kron(sparse.identity(6**2,format='csr'),
                        sparse.kron(g5_operator,
                                    sparse.identity(6**3,format='csr')))))
    plaquette1 /= 6

    return plaquette1

def plaquette2():
    plaquette2 = sparse.csr_matrix((6**8,6**8))

    for i in range(6):                                      # sum over all elements in the group
        h = np.zeros(6)
        h[i] = 1

        g2_operator = sparse.csr_matrix(right_mult(inv @ h))
        g3_operator = sparse.csr_matrix(right_mult(inv @ h))
        g6_operator = sparse.csr_matrix(right_mult(inv @ h))

        plaquette2 += sparse.kron(sparse.identity(6,format='csr'),
                        sparse.kron(g2_operator,
                        sparse.kron(g3_operator,
                        sparse.kron(sparse.identity(6**2,format='csr'),
                        sparse.kron(g6_operator,
                                    sparse.identity(6**2,format='csr'))))))
    plaquette2 /= 6

    return plaquette2

def plaquette3():
    plaquette3 = sparse.csr_matrix((6**8,6**8))

    for i in range(6):                                      # sum over all elements in the group
        h = np.zeros(6)
        h[i] = 1

        g3_operator = sparse.csr_matrix(left_mult(h))
        g4_operator = sparse.csr_matrix(left_mult(h))
        g7_operator = sparse.csr_matrix(left_mult(h))

        plaquette3 += sparse.kron(sparse.identity(6**2,format='csr'),
                        sparse.kron(g3_operator,
                        sparse.kron(g4_operator,
                        sparse.kron(sparse.identity(6**2,format='csr'),
                        sparse.kron(g7_operator,
                                    sparse.identity(6,format='csr'))))))

    plaquette3 /= 6

    return plaquette3

def plaquette4():
    plaquette4 = sparse.csr_matrix((6**8,6**8))

    for i in range(6):                                      # sum over all elements in the group
        h = np.zeros(6)
        h[i] = 1

        g1_operator = sparse.csr_matrix(right_mult(inv @ h))
        g4_operator = sparse.csr_matrix(right_mult(inv @ h))
        g8_operator = sparse.csr_matrix(right_mult(inv @ h))

        plaquette4 += sparse.kron(g1_operator,
                        sparse.kron(sparse.identity(6**2,format='csr'),
                        sparse.kron(g4_operator,
                        sparse.kron(sparse.identity(6**3,format='csr'),
                                    g8_operator))))

    plaquette4 /= 6

    return plaquette4

def outer_plaquette():

    outer_plaquette = sparse.csr_matrix((6**8,6**8))

    for i in range(6):                                      # sum over all elements in the group
        h = np.zeros(6)
        h[i] = 1

        g5_operator = sparse.csr_matrix(right_mult(inv @ h))
        g6_operator = sparse.csr_matrix(left_mult(h))
        g7_operator = sparse.csr_matrix(right_mult(inv @ h))
        g8_operator = sparse.csr_matrix(left_mult(h))

        outer_plaquette += sparse.kron(sparse.identity(6**4,format='csr'),
                            sparse.kron(g5_operator,
                            sparse.kron(g6_operator,
                            sparse.kron(g7_operator,
                                        g8_operator))))

    outer_plaquette /= 6

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
    perturbation = sparse.csr_matrix((6**8,6**8))


    for h in conjugacy_class[element]:

        if side == 'l':
            multiplier = sparse.csr_matrix(left_mult(h))
        elif side == 'r':
            multiplier = sparse.csr_matrix(right_mult(h))

        perturbation += sparse.kron(sparse.identity(6**2,format='csr'),
                        sparse.kron(multiplier,
                                    sparse.identity(6**5,format='csr')))

    return perturbation

def perturbation1(element,side='l'):
    perturbation = sparse.csr_matrix((6**8,6**8))


    for h in conjugacy_class[element]:

        if side == 'l':
            multiplier = sparse.csr_matrix(left_mult(h))
        elif side == 'r':
            multiplier = sparse.csr_matrix(right_mult(h))

        perturbation += sparse.kron(multiplier, sparse.identity(6**7,format='csr'))

    return perturbation

def remote_perturbation24(element):

    perturbation = sparse.csr_matrix((6**8,6**8))

    for i in range(6):                              # Loop over all values g4 can take and project
        g4 = np.zeros(6)
        g4[i] = 1

        ribbon = sparse.csr_matrix((6**8,6**8))

        projector = sparse.kron(sparse.identity(6**3,format='csr'),
                    sparse.kron(sparse.csr_matrix(np.outer(g4,g4)),
                                sparse.identity(6**4,format='csr')))

        for h in conjugacy_class[element]:

            multiplier1 = sparse.csr_matrix(right_mult(g4) @ right_mult(h) @ right_mult(inv @ g4))

            multiplier3 = sparse.csr_matrix(left_mult(inv @ h))

            ribbon += sparse.kron(multiplier1,
                        sparse.kron(sparse.identity(6,format='csr'),
                        sparse.kron(multiplier3,
                                    sparse.identity(6**5,format='csr'))))

        ribbon /= np.sqrt(len(conjugacy_class[element]))
        perturbation += ribbon @ projector

    return perturbation

def remote_perturbation35(element):

    perturbation = sparse.csr_matrix((6**8,6**8))

    for i in range(6):                              # Loop over all values g4 can take and project
        g1 = np.zeros(6)
        g1[i] = 1

        ribbon = sparse.csr_matrix((6**8,6**8))

        projector = sparse.kron(sparse.csr_matrix(np.outer(g1,g1)),
                                sparse.identity(6**7,format='csr'))

        for h in conjugacy_class[element]:

            multiplier2 = sparse.csr_matrix(left_mult(g1) @ left_mult(inv @ h) @ left_mult(inv @ g1))

            multiplier4 = sparse.csr_matrix(right_mult(h))

            ribbon +=   sparse.kron(sparse.identity(6,format='csr'),
                        sparse.kron(multiplier2,
                        sparse.kron(sparse.identity(6,format='csr'),
                        sparse.kron(multiplier4,
                                    sparse.identity(6**4,format='csr')))))

        ribbon /= np.sqrt(len(conjugacy_class[element]))
        perturbation += ribbon @ projector

    return perturbation

def commutation():

    classes = ['R','m']

    for blue in classes:
        for red in classes:

            a = remote_perturbation24(blue)@remote_perturbation35(red)@GS
            b = remote_perturbation35(red)@remote_perturbation24(blue)@GS

            print('Blue: ',blue,'Red: ',red, 'Inner product: ',np.dot(a,b))








