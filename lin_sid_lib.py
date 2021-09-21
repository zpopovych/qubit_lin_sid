''' Basic Linear System Identification function and testing tools '''
''' block_hankel'''
''' identify_linear'''
''' simulate_linear'''

import numpy as np
from scipy.integrate import ode
from scipy.linalg import svd, pinv, logm
from collections import namedtuple
import h5py

linear_model = namedtuple('linear_model', ['Ac', 'C', 'x0'])

def block_hankel(first_column, last_row):
    """
    Form block Hankel matrix from first_column and last_row.
    last_row[0] is ignored; the last row of the returned matrix is [first_column[-1], last_row[1:]].

    Input can be matrix or vector
    Hankel matrix is build by the elements of the first index (dimension) of the argument,
    thus to build Hankel of vectors input should be 2D,
    and to build Hankel of matrices input should be 3D
    """
    blocks = [[]]

    # fill-up with the upper left triangular with first_column values
    for entry in first_column:
        for row in blocks:
            row.append(entry)
        blocks.append([])

    blocks.pop()

    # fill-up with the low right triangular with last_row values
    for row in blocks:
        row.extend(last_row[1:])

    # trim the blocks such that it is a matrix
    blocks = [row[:len(last_row)] for row in blocks]

    return np.block(blocks)


def identify_linear(Y, dt, rank:int=None, precision = 1e-9):
    '''
    Time should be the first index of Y,
    2nd index - rows (m - outputs)
    3rd index - columns (r - inputs)
    '''
    N, m, r = Y.shape

    #print('Processing series of length='+ str(N) + ' of ' + str(m) + 'x' + str(r) + '-matrices.')

    q = round(N/2)

    #print("round(N/2)=", q)

    H = block_hankel(Y[:q], Y[q - 1:])

    #print('Hankel done. Hankel shape:', H.shape)

    U, Sigma, V_T = svd(H, full_matrices=False, overwrite_a=True)

    #print('SVD done. \n Sigma shape:', Sigma.shape)
    #print('U.shape = ', U.shape)
    #print('V_T.shape = ', V_T.shape)
    #print('Sigma=', Sigma)

    sqrt_sigma = np.sqrt(Sigma)

    #print('Square root of sigma done.')

    # I found the internally balanced realization to work the best in quantum case
    U *= sqrt_sigma[None, ...]
    V_T *= sqrt_sigma[..., None]

    # estimate the rank if it is not given
    S = np.abs(Sigma)

    #print(np.abs(S / S.max()))
    #print(np.argmin(np.abs(S / S.max())))

    rank = rank if rank else np.argmin(np.abs(S / S.max() - precision)) # was 1e-3 try
    #rank =2

    #print('Estimated rank:', rank)

    # C = the first m rows of U

    C = U[:m, :rank] # m - number of outputs, r - number of inputs

    #print('C.shape: ', C.shape)

    # We cut out last m-rows of U
    U_up = U[:-m, ]
    # We cut out first m-rows of U
    U_down = U[m:, ]

    '''
    Ac_reconstructed = logm(
        orthogonal_procrustes(U1_up, U1_down)[0][:rank, :rank]
        if enforce_orthogonal
        else lstsq(U1_up, U1_down)[0][:rank, :rank]
    ) / dt
    '''

    A = pinv(U_up) @ U_down

    #print('A.shape: ', A.shape)

    try:
        Ac = logm(A) / dt
        Ac = Ac[:rank, :rank]
    except:
        Ac = None
        raise("Matrix logarithm failed!!!")

    x0 = pinv(U) @ H
    x0 = x0[:rank, 0]

    #print('Initial state x0 identified:', x0)

    return linear_model(Ac = Ac, C=C, x0 = x0), Sigma

def simulate_linear(model: linear_model,  times: np.ndarray):

    integrator_kind = 'zvode'
    solver = ode( lambda t, x, Ac : Ac @ x ).set_integrator(integrator_kind)
    solver.set_initial_value(model.x0.reshape(-1), times[0]).set_f_params(model.Ac)
    # save the trajectory
    x = [solver.y]
    x.extend(
        solver.integrate(t) for t in times[1:]
    )

    C = np.array(model.C)
    res = C @ np.array(x).T
    res = res.T
    res.shape += 1,

    return res

def save_model(f_mod_name, model):
    Ac = model.Ac
    C = model.C
    x0 = model.x0
    with h5py.File(f_mod_name, 'a') as f:
        f.require_dataset('Ac', Ac.shape, dtype='complex', data=Ac)
        f.require_dataset('C', C.shape, dtype='complex', data=C)
        f.require_dataset('x0', x0.shape, dtype='complex', data=x0)
    return


def load_model(f_mod_name):
    with h5py.File(f_mod_name, 'r') as f:
        Ac = np.array(f['Ac'])
        C = np.array(f['C'])
        x0 = np.array(f['x0'])
        mod = linear_model(Ac, C, x0)
    return mod

def sigma(Y):

    N, m, r = Y.shape
    print(
        'Processing SVD of block-hankel of series of length=' + str(N) + ' of ' + str(m) + 'x' + str(r) + '-matrices.')
    q = round(N / 2)
    print('Hankel...')
    H = block_hankel(Y[:q], Y[q - 1:])
    print('SVD...')
    U, Sigma, V_T = np.linalg.svd(H, full_matrices=False)

    return Sigma


def hankel_and_svd(Y):
    '''
    Time should be the first index of Y,
    2nd index - rows (m - outputs)
    3rd index - columns (r - inputs)
    '''
    N, m, r = Y.shape

    q = round(N/2)

    H = block_hankel(Y[:q], Y[q - 1:])

    U, Sigma, V_T = svd(H, full_matrices=False, overwrite_a=True)

    sqrt_sigma = np.sqrt(Sigma)

    # I found the internally balanced realization to work the best in quantum case
    U *= sqrt_sigma[None, ...]
    V_T *= sqrt_sigma[..., None]

    # estimate the rank if it is not given
    S = np.abs(Sigma)

    #if U.shape[1] > 50: U = U[:m, :50]

    return H, U, S, m

def identify_from_hankel_and_svd(H, U, m, rank, dt):

    C = U[:m, :rank] # m - number of outputs, r - number of inputs
    # We cut out last m-rows of U
    U_up = U[:-m, ]
    # We cut out first m-rows of U
    U_down = U[m:, ]
    A = pinv(U_up) @ U_down

    try:
        Ac = logm(A) / dt
        Ac = Ac[:rank, :rank]
    except:
        Ac = None
        raise("Matrix logarithm failed!!!")

    x0 = pinv(U) @ H
    x0 = x0[:rank, 0]

    return linear_model(Ac = Ac, C=C, x0 = x0)