import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

'''
The code denotes the 6 edges as g1, g2, etc., and similarly for vertices and plaquettes. In a different document you can find how I have labelled and oriented all the edges, vertices and plaquettes.

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

# Below the functions that generate all the matrices representing vertex operators. They are sums over projectors onto the allowed sets of group elements that satisfy the vertex condition. Labelling of edges in other file.

def vertex1_check(g1,g3,g4,g6):                             # for the four-edge vertex, checks if
    product = left_mult(inv @ g6) @ (inv @ g3)              # vertex condition is met
    product = left_mult(g4) @ product
    product = left_mult(g1) @ product
    return np.array_equal(product, np.array([1,0,0,0,0,0,0,0]))


def vertex1():

    vertex1 = sparse.csr_matrix((8**6,8**6))

    for i in range(8):                                      # The for loops go over all possible sets
        g1 = np.zeros(8)                                    # of four group elements
        g1[i] = 1

        for j in range(8):
            g3 = np.zeros(8)
            g3[j] = 1

            for k in range(8):
                g4 = np.zeros(8)
                g4[k] = 1

                for l in range(8):
                    g6 = np.zeros(8)
                    g6[l] = 1

                    if vertex1_check(g1,g3,g4,g6) == True:

                    # These then produce projectors as outer products and use matrix kronecker products
                    # to stick them together to act on all 6 edges.

                        projector1 = sparse.csr_matrix(np.outer(g1,g1))
                        projector3 = sparse.csr_matrix(np.outer(g3,g3))
                        projector4 = sparse.csr_matrix(np.outer(g4,g4))
                        projector6 = sparse.csr_matrix(np.outer(g6,g6))

                        projector = sparse.kron(projector1,
                                    sparse.kron(sparse.identity(8,format='csr'),
                                    sparse.kron(projector3,
                                    sparse.kron(projector4,
                                    sparse.kron(sparse.identity(8,format='csr'),
                                                projector6)))))

                        vertex1 = vertex1 + projector


    return vertex1

# The vertex condition for the other four vertices (each with two edges) is very simple. The projectors are then produced just as above.

def vertex2():
    vertex2 = sparse.csr_matrix((8**6,8**6))

    for i in range(8):
        g = np.zeros(8)
        g[i] = 1

        projector_g = sparse.csr_matrix(np.outer(g,g))


        projector = sparse.kron(projector_g,
                    sparse.kron(projector_g,
                                sparse.identity(8**4,format='csr')))
        vertex2 += projector

    return vertex2

def vertex3():
    vertex3 = sparse.csr_matrix((8**6,8**6))

    for i in range(8):
        g = np.zeros(8)
        g[i] = 1

        projector_g = sparse.csr_matrix(np.outer(g,g))

        projector = sparse.kron(sparse.identity(8,format='csr'),
                    sparse.kron(projector_g,
                    sparse.kron(projector_g,
                                sparse.identity(8**3,format='csr'))))
        vertex3 += projector

    return vertex3

def vertex4():
    vertex4 = sparse.csr_matrix((8**6,8**6))

    for i in range(8):
        g = np.zeros(8)
        g[i] = 1

        projector_g = sparse.csr_matrix(np.outer(g,g))

        projector = sparse.kron(sparse.identity(8**3,format='csr'),
                    sparse.kron(projector_g,
                    sparse.kron(projector_g,
                                sparse.identity(8,format='csr'))))
        vertex4 += projector

    return vertex4

def vertex5():
    vertex5 = sparse.csr_matrix((8**6,8**6))

    for i in range(8):
        g = np.zeros(8)
        g[i] = 1

        projector_g = sparse.csr_matrix(np.outer(g,g))

        projector = sparse.kron(sparse.identity(8**4,format='csr'),
                    sparse.kron(projector_g,
                            projector_g))
        vertex5 += projector

    return vertex5

# The plaquette operators are also straightforward. Depending on orientation either left-multiplying by h (for arrows oriented counter-clockwise, plaquette 1 in this case) or right-multiplying by the inverse of h (plaquette 2). For the outer plaquette it is a combination of the two.

# Then just sum over all h in the group

def plaquette1():
    plaquette1 = sparse.csr_matrix((8**6,8**6))

    for i in range(8):                                      # sum over all elements in the group
        h = np.zeros(8)
        h[i] = 1

        g1_operator = sparse.csr_matrix(left_mult(h))
        g2_operator = sparse.csr_matrix(left_mult(h))
        g3_operator = sparse.csr_matrix(left_mult(h))

        plaquette1 += sparse.kron(g1_operator,
                        sparse.kron(g2_operator,
                        sparse.kron(g3_operator,
                                    sparse.identity(8**3,format='csr'))))
    plaquette1 /= 8

    return plaquette1

def plaquette2():
    plaquette2 = sparse.csr_matrix((8**6,8**6))

    for i in range(8):
        h = np.zeros(8)
        h[i] = 1

        g4_operator = sparse.csr_matrix(right_mult(inv @ h))
        g5_operator = sparse.csr_matrix(right_mult(inv @ h))
        g6_operator = sparse.csr_matrix(right_mult(inv @ h))

        plaquette2 += sparse.kron(sparse.identity(8**3,format='csr'),
                        sparse.kron(g4_operator,
                        sparse.kron(g5_operator,
                                    g6_operator)))
    plaquette2 /= 8

    return plaquette2

def outer_plaquette():
    outer_plaquette = sparse.csr_matrix((8**6,8**6))

    for i in range(8):
        h = np.zeros(8)
        h[i] = 1

        g1_operator = sparse.csr_matrix(right_mult(inv @ h))
        g2_operator = sparse.csr_matrix(right_mult(inv @ h))
        g3_operator = sparse.csr_matrix(right_mult(inv @ h))
        g4_operator = sparse.csr_matrix(left_mult(h))
        g5_operator = sparse.csr_matrix(left_mult(h))
        g6_operator = sparse.csr_matrix(left_mult(h))

        outer_plaquette +=  sparse.kron(g1_operator,
                            sparse.kron(g2_operator,
                            sparse.kron(g3_operator,
                            sparse.kron(g4_operator,
                            sparse.kron(g5_operator,
                                        g6_operator)))))

    outer_plaquette /= 8

    return outer_plaquette

# These lines just call all the functions to actually generate the matrices, which are then summed to produce the Hamiltonian.

vertex1 = vertex1()
vertex2 = vertex2()
vertex3 = vertex3()
vertex4 = vertex4()
vertex5 = vertex5()
plaquette1 = plaquette1()
plaquette2 = plaquette2()
outer_plaquette = outer_plaquette()


Hamiltonian = -(vertex1 + vertex2 + vertex3 + vertex4 + vertex5 + plaquette1 + plaquette2 + outer_plaquette)


# Findings: unique ground state, eigenvalue -8

# 61 lowest excited states, eigenvalue -6.

# These lines extract the ground state to be used in later stages of calculation.

GS = eigsh(Hamiltonian, k=1, which='SA')[1][:,0]
GS[np.abs(GS)<1e-12] = 0

# The following two functions are used to easily assess the effects of any perturbing operators.

# The first function finds the eigenvalue given a matrix and an eigenstate. The second one loops over the Hamiltonian and all its components and gives eigenvalues for each, using the dictionary.

matrices = {'vertex 1': vertex1,
            'vertex 2': vertex2,
            'vertex 3': vertex3,
            'vertex 4': vertex4,
            'vertex 5': vertex5,
            'plaquette 1': plaquette1,
            'plaquette 2': plaquette2,
            'outer plaquette': outer_plaquette,
            'Hamiltonian': Hamiltonian}

def find_EV(matrix, state):

    for i in range(-8,2):
        if np.allclose(matrix @ state, i*state, atol=1e-10,rtol=0) == True:
            return i
    return 'no eigenvalue found'

def EV_checks(state):
    for key in matrices:
        print(key, ': ', find_EV(matrices[key],state))


# These matrices give perturbations given a group element in string format. They either premultiply or postmultiply an edge by a conjugacy class. There is one for edge 6 and one for edge 1 (see document with labelling).

def perturbation6(element,side='l'):
    perturbation = sparse.csr_matrix((8**6,8**6))

    for h in conjugacy_class(element):

        if side == 'l':
            multiplier = sparse.csr_matrix(left_mult(h))

        elif side == 'r':
            multiplier = sparse.csr_matrix(right_mult(h))

        perturbation += sparse.kron(sparse.identity(8**5,format='csr'),multiplier)

    return perturbation

def perturbation1(element,side='l'):
    perturbation = sparse.csr_matrix((8**6,8**6))

    for h in conjugacy_class(element):

        if side == 'l':
            multiplier = sparse.csr_matrix(left_mult(h))

        elif side == 'r':
            multiplier = sparse.csr_matrix(right_mult(h))

        perturbation += sparse.kron(multiplier,sparse.identity(8**5,format='csr'))

    return perturbation

def remote_perturbation25(element):

    perturbation = sparse.csr_matrix((8**6,8**6))

    for i in range(8):                              # Loop over all values g4 can take and project
        g4 = np.zeros(8)
        g4[i] = 1

        ribbon = sparse.csr_matrix((8**6,8**6))

        projector = sparse.kron(sparse.identity(8**3,format='csr'),
                    sparse.kron(sparse.csr_matrix(np.outer(g4,g4)),
                                sparse.identity(8**2,format='csr')))

        for h in conjugacy_class(element):

            multiplier1 = sparse.csr_matrix(right_mult(inv @ g4) @ right_mult(h) @ right_mult(g4))

            multiplier6 = sparse.csr_matrix(right_mult(h))

            ribbon +=   sparse.kron(multiplier1,
                        sparse.kron(sparse.identity(8**4,format='csr'),
                                    multiplier6))

        ribbon /= np.sqrt(len(conjugacy_class(element)))
        perturbation += ribbon @ projector

    return perturbation

def remote_perturbation34(element):

    perturbation = sparse.csr_matrix((8**6,8**6))

    for i in range(8):                              # Loop over all values g1 can take and project
        g6 = np.zeros(8)
        g6[i] = 1

        ribbon = sparse.csr_matrix((8**6,8**6))

        projector = sparse.kron(sparse.identity(8**5,format='csr'),
                                sparse.csr_matrix(np.outer(g6,g6)))

        for h in conjugacy_class(element):

            multiplier3 = sparse.csr_matrix(right_mult(h))
            multiplier4 = sparse.csr_matrix(right_mult(g6) @ right_mult(h) @ right_mult(inv @ g6))

            ribbon +=   sparse.kron(sparse.identity(8**2,format='csr'),
                        sparse.kron(multiplier3,
                        sparse.kron(multiplier4,
                                    sparse.identity(8**2,format='csr'))))

        ribbon /= np.sqrt(len(conjugacy_class(element)))
        perturbation += ribbon @ projector

    return perturbation

def commutation():

    classes = ['i','j','k']

    for blue in classes:
        for red in classes:

            a = remote_perturbation25(blue)@remote_perturbation34(red)@GS
            b = remote_perturbation34(red)@remote_perturbation25(blue)@GS

            print('Blue: ',blue,'Red: ',red, 'Inner product: ',np.dot(a,b))
















