# These first two create vertex defects by multiplying by a single element. However, this also creates a plaquette defect on one side of the edge.

def simple_perturbation6(element):
    perturbation = sparse.csr_matrix((46656,46656))
    element = vector(element)

    multiplier = sparse.csr_matrix(right_mult(element))
    perturbation += sparse.kron(sparse.identity(6**5,format='csr'),
                                    multiplier)
    return perturbation

def simple_perturbation1(element):
    perturbation = sparse.csr_matrix((46656,46656))
    element = vector(element)

    multiplier = sparse.csr_matrix(right_mult(element))
    perturbation += sparse.kron(multiplier, sparse.identity(6**5,format='csr'))
    return perturbation



# This functions projects a state onto specific states of vector g3, written in preparation for making a two-edge ribbon operator.

def projector(g3):
    projector = sparse.csr_matrix((46656,46656))
    g3 = vector(g3)

    projector += sparse.kron(sparse.identity(36,format='csr'),
                sparse.kron(sparse.csr_matrix(np.outer(g3,g3)),
                            sparse.identity(216,format='csr')))

    return projector


# This function inverts the effect of the conjugation class perturbation numerically. It works for (R,R2), but the matrix associated with the reflections is singular and gives an error.

def annihilate1(element):

    element = vector(element)
    multiplier = np.zeros((6,6))

    for i in range(6):
        p = np.zeros(6)
        p[i] = 1


        multiplier += left_mult(p) @ left_mult(element) @ left_mult(inv @ p)

    multiplier = sparse.csr_matrix(np.linalg.inv(multiplier))

    annihilate = sparse.kron(multiplier,sparse.identity(6**5,format='csr'))


    return annihilate
