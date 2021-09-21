import numpy as np
from lin_sid_lib import *
from numpy.linalg import eigvals

def vec_of_rho(rho):
    assert np.max(np.abs(rho[:, 0, 1].real - rho[:, 1, 0].real)) < 1e-3
    assert np.max(np.abs(rho[:, 0, 1].imag + rho[:, 1, 0].imag)) < 1e-3
    assert np.max(np.abs(rho[:, 0, 0].real + rho[:, 1, 1].real)-1) < 1e-3
    assert np.max(np.abs(rho[:, 0, 0].imag)) < 1e-3
    assert np.max(np.abs(rho[:, 1, 1].imag)) < 1e-3

    g = rho[:, 0, 0].real
    # e = rho[:, 1, 1].real
    r = rho[:, 0, 1].real
    i = rho[:, 0, 1].imag

    return np.vstack((g, r, i)).T

def y_of_rho (rho0, rho1, rhoX, rhoY):
    assert (rho0.shape[0] == rho1.shape[0] == rhoX.shape[0] == rhoY.shape[0])
    y0 = vec_of_rho(rho0)  # excited
    y1 = vec_of_rho(rho1)  # ground
    yX = vec_of_rho(rhoX)
    yY = vec_of_rho(rhoY)

    Y = np.hstack((y0, y1, yX, yY))
    Y.shape += 1,

    return Y

def rho_of_y(y):
    rho0 = rho_of_vec(y[:, 0:3]) # excited init state
    rho1 = rho_of_vec(y[:, 3:6]) #  ground init state
    rhoX = rho_of_vec(y[:, 6:9])
    rhoY = rho_of_vec(y[:, 9:12])

    return rho0, rho1, rhoX, rhoY


def rho_of_vec (vec):
    vec.shape = (vec.shape[0], 3)
    rho = np.empty(shape=(vec.shape[0], 2, 2), dtype='complex')
    rho[:, 0, 0] = vec[:, 0]            # excited state population
    rho[:, 1, 1] = 1 - vec[:, 0]        # ground state population
    rho[:, 0, 1] = vec[:, 1] + 1j*vec[:, 2] # coherence
    rho[:, 1, 0] = vec[:, 1] - 1j * vec[:, 2] # conjugate of coherence

    return rho

def check_rho(rho, prec=1e-6):
    #from np.linalg import eigvals
    from scipy.linalg import eigvalsh

    assert np.max(np.abs(rho[:, 0, 1].real - rho[:, 1, 0].real)) < prec # real part of coherence is the same
    assert np.max(np.abs(rho[:, 0, 1].imag + rho[:, 1, 0].imag)) < prec # imag part of coherence corresponds
    assert np.max(np.abs(rho[:, 0, 0].real + rho[:, 1, 1].real) - 1) < prec # real Trace  = 1
    assert np.max(np.abs(rho[:, 0, 0].real)) <= 1. # diagonal elements < 1
    assert np.max(np.abs(rho[:, 1, 1].real)) <= 1. # diagonal elements < 1
    assert np.max(np.abs(rho[:, 0, 0].real)) >= 0. # diagonal elements positive
    assert np.max(np.abs(rho[:, 1, 1].real)) >= 0. # diagonal elements positive
    assert np.max(np.abs(rho[:, 0, 0].imag)) < prec # diagonal elements real
    assert np.max(np.abs(rho[:, 1, 1].imag)) < prec # diagonal elements real

    max_purity = np.max(purity_of_rho(rho))
    if max_purity > 1+prec:
        print('Max of purity:', max_purity)

    neg_eig_point_count = 0
    complex_eig_point_count = 0
    bad_purity_points_count = 0

    min_eig = 0
    len_of_rho = rho.shape[0]
    purity = purity_of_rho(rho)
    for i in range(0, len_of_rho):
        if purity[i] > 1 + prec:
            bad_purity_points_count +=1

        if np.max(np.abs(eigvals(rho[i]).imag)) > prec:
            complex_eig_point_count +=1

        cur_min_eig = np.min(eigvals(rho[i]).real)
        if cur_min_eig < -prec:
            neg_eig_point_count=+1
            if cur_min_eig  < min_eig:
                min_eig = cur_min_eig
                min_eig_pnt = i

    if neg_eig_point_count > 0 or complex_eig_point_count > 0 or bad_purity_points_count > 0:
        print('Number of points with bad purity:', bad_purity_points_count)
        print('Number of points with complex eigenvalues:', complex_eig_point_count)
        print('Number of points with negative eigenvalues:', neg_eig_point_count)
        print('Bigest negative eigenvalue:', min_eig)
        print('at point #', min_eig_pnt, end=" ")
        print(' of ', len_of_rho)


def propagate_rho(init_rho, U, nsteps): # U - is propagator
    rho_prop = [init_rho]

    for _ in range(nsteps - 1):
        rho_prop.append(
            U @ rho_prop[-1] @ U.conj().T
        )

    return np.array(rho_prop)

def evolve_rho(init_rho, model, t, exact_init_states = False):
    assert(init_rho[0, 0].real + init_rho[1,1].real - 1. < 1e-3)
    assert(init_rho[0, 0].imag < 1e-3)
    assert (init_rho[1, 1].imag < 1e-3)

    #kg = init_rho[0,0].real - init_rho[0,1].real + init_rho[0,1].imag
    #ke = 1 - init_rho[0,0].real - init_rho[0,1].real + init_rho[0,1].imag
    #kx = 2 * init_rho[0,1].real
    #ky = -2 * init_rho[0,1].imag
    #print(kg, ke, kx, ky)


    y = simulate_linear(model, t)

    if exact_init_states:
        y[0] = np.array([0., 0., 0., 1., 0., 0., .5, .5, 0., .5, 0., .5]).reshape((12,1))

    rho0, rho1, rhoX, rhoY = rho_of_y(y)

    #rho0 = rho_of_vec(y[:, 0:3]) # excited
    #rho1 = rho_of_vec(y[:, 3:6]) # ground
    #rhoX = rho_of_vec(y[:, 6:9])
    #rhoY = rho_of_vec(y[:, 9:12])

    evolution_of_rho = rho_01XY(init_rho, rho0, rho1, rhoX, rhoY)

    #evolution_of_rho = kg*rho0 + ke*rho1 + kx*rhoX + ky*rhoY

    return evolution_of_rho

def purity_of_rho(rho_array):
    return([np.trace(rho @ rho) for  rho in rho_array])

def plot_rho(rho, title=r'Density matrix $\rho$ :'):
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 4, 1)
    plt.title(title + '\n Exited state'+ r' $\rho_{00}$')
    plt.plot(rho[:, 0, 0].real)

    plt.subplot(1, 4, 2)
    plt.title('Ground state'+ r' $\rho_{11}$')
    plt.plot(rho[:, 1, 1].real)

    plt.subplot(1, 4, 3)
    plt.title('Coherence.real'+ r' Re($\rho_{10}$)')
    plt.plot(rho[:, 1, 0].real)

    plt.subplot(1, 4, 4)
    plt.title('Coherence.imag'+ r' Im($\rho_{10}$)')
    plt.plot(rho[:, 1, 0].imag)
    plt.show()

def plot2rho(rho1, rho2, title=r'Density matrix $\rho$ :',l1='1', l2='2', mark='*'):
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 4, 1)
    plt.title(title + '\n Exited  state'+ r' $\rho_{00}$')
    plt.plot(rho1[:, 0, 0].real, marker=mark)
    plt.plot(rho2[:, 0, 0].real, marker=mark)

    plt.subplot(1, 4, 2)
    plt.title('Ground state'+ r' $\rho_{11}$')
    plt.plot(rho1[:, 1, 1].real, marker=mark)
    plt.plot(rho2[:, 1, 1].real, marker=mark)


    plt.subplot(1, 4, 3)
    plt.title('Coherence.real' + r' Re($\rho_{10}$)')
    plt.plot(rho1[:, 0, 1].real, marker=mark)
    plt.plot(rho2[:, 0, 1].real, marker=mark)

    plt.subplot(1, 4, 4)
    plt.title('Coherence.imag' + r' Im($\rho_{10}$)')
    plt.plot(rho1[:, 0, 1].imag, label=l1, marker=mark)
    plt.plot(rho2[:, 0, 1].imag, label=l2, marker=mark)
    plt.legend()
    plt.show()

def plot_rho_vs_rho_pur (rho1, rho2, title=r'Density matrix $\rho$ :', l1='1', l2='2', mark='*'):

    fig, [ax1, ax2] = plt.subplots(2,1,figsize=(10, 10))

    ax1.set_title(title + '\n Exited  state'+ r' $\rho_{00}$')
    ax1.plot(rho1[:, 0, 0].real, marker=mark,  label=l1, c='blue')
    ax1.plot(rho2[:, 0, 0].real, marker=mark, label=l2, c='orange')

    line11, = ax1.plot(rho1[:, 0, 1].real, c='violet')
    line11.set_dashes([1, 5, 10, 5]) # 1pt line, 5pt break, 10pt line, 5pt break
    line12, = ax1.plot(rho2[:, 0, 1].real, c='red', label='coh.real')
    line12.set_dashes([1, 5, 10, 5])

    line21, = ax1.plot(rho1[:, 0, 1].imag, c='violet')
    line21.set_dashes([1, 3, 3, 0])
    line22, = ax1.plot(rho2[:, 0, 1].imag, c='red', label='coh.imag')
    line22.set_dashes([1, 3, 3, 0])
    ax1.legend()

    ax2.set_title('Purity')
    ax2.plot(purity_of_rho(rho1), marker=mark, c='blue')
    ax2.plot(purity_of_rho(rho2), marker=mark, c='orange')

    return fig

def plot_3rho(rho1, rho2, rho3, title=r'Density matrix $\rho$ :', l1='1', l2='2', l3='3', mark='*'):

    fig, [ax1, ax2] = plt.subplots(2,1,figsize=(10, 10))

    ax1.set_title(title + '\n Exited  state'+ r' $\rho_{00}$')
    ax1.plot(rho1[:, 0, 0].real, marker=mark,  label=l1, c='blue')
    ax1.plot(rho2[:, 0, 0].real, marker=mark, label=l2, c='orange')
    ax1.plot(rho3[:, 0, 0].real, marker=mark, label=l3, c='green')

    line11, = ax1.plot(rho1[:, 0, 1].real, c='violet')
    line11.set_dashes([1, 5, 10, 5]) # 1pt line, 5pt break, 10pt line, 5pt break
    line12, = ax1.plot(rho2[:, 0, 1].real, c='red', label='coh.real')
    line12.set_dashes([1, 5, 10, 5])

    line21, = ax1.plot(rho1[:, 0, 1].imag, c='violet')
    line21.set_dashes([1, 3, 3, 0])
    line22, = ax1.plot(rho2[:, 0, 1].imag, c='red', label='coh.imag')
    line22.set_dashes([1, 3, 3, 0])

    line31, = ax1.plot(rho3[:, 0, 1].imag, c='violet')
    line31.set_dashes([1, 1, 1, 0])
    line32, = ax1.plot(rho3[:, 0, 1].imag, c='red', label='coh.imag')
    line32.set_dashes([1, 1, 1, 0])

    ax1.legend()

    ax2.set_title('Purity')
    ax2.plot(purity_of_rho(rho1), marker=mark, c='blue')
    ax2.plot(purity_of_rho(rho2), marker=mark, c='orange')
    ax2.plot(purity_of_rho(rho3), marker=mark, c='green')

    return fig


def plot_purity(rho):
    plt.figure(figsize=(10, 10))
    plt.title('Purity of ' + r' $\rho$')
    plt.plot(purity_of_rho(rho))
    plt.show()

def plot_2purity(rho1, rho2, l1='1', l2='2'):
    plt.figure(figsize=(10, 10))
    plt.title('Purity of ' + r' $\rho$')
    plt.plot(purity_of_rho(rho1), label=l1)
    plt.plot(purity_of_rho(rho2), label=l2)
    plt.legend()
    plt.show()

def plot_sigma(y):
    s = sigma(y)
    plt.title(r'$\frac{\Sigma_i}{ \mathrm{max}( \Sigma )}$')
    plt.semilogy(s / s.max(), "-+")
    plt.show()

def rho_01XY(rho_i, rho0, rho1, rhoX, rhoY):
    # rho_i initial density maxrix
    prec = 1e-3

    assert(rho_i[0, 0].real + rho_i[1,1].real - 1. < prec)
    assert(rho_i[0, 0].imag < prec)
    assert(rho_i[1, 1].imag < prec)

    # rho[:,0,0] excited state population
    # rho[:,1,1] ground state population
    # rho[:,0,1] coherence
    # rho[:,1,0] complex conjugate of coherence

    ke = rho_i[0, 0].real - rho_i[1, 0].real + rho_i[1, 0].imag
    kg = rho_i[1, 1].real - rho_i[1, 0].real + rho_i[1, 0].imag
    # kg = 1 - rho_i[0,0].real - rho_i[0,1].real + rho_i[0,1].imag

    #kx =  2 * rho_i[0,1].real
    #ky = -2 * rho_i[0,1].imag

    kx =  2 * rho_i[1,0].real
    ky =  - 2 * rho_i[1,0].imag

    rho_01XY = rho0 * kg + rho1 * ke + rhoX * kx + rhoY * ky

    return rho_01XY

def plot_rho_vs_rho (rho1, rho2, title=r'Density matrix $\rho$ :', l1='1', l2='2', mark='*'):

    fig, ax1 = plt.subplots(1,1,figsize=(10, 5))
    #Excited state r' $\rho_{00}$'

    ax1.set_title(title)
    ax1.plot(rho1[:, 0, 0].real, marker=mark,  label=l1 + r' $\rho_{00}$', c='blue')
    ax1.plot(rho2[:, 0, 0].real, marker=mark, label=l2 + r' $\rho_{00}$', c='orange')

    line11, = ax1.plot(rho1[:, 0, 1].real, c='violet')
    line11.set_dashes([1, 5, 10, 5]) # 1pt line, 5pt break, 10pt line, 5pt break
    line12, = ax1.plot(rho2[:, 0, 1].real, c='red', label= r' Re[$\rho_{01}$]')
    line12.set_dashes([1, 5, 10, 5])

    line21, = ax1.plot(rho1[:, 0, 1].imag, c='violet')
    line21.set_dashes([1, 3, 3, 0])
    line22, = ax1.plot(rho2[:, 0, 1].imag, c='red', label=r' Im[$\rho_{01}$]')
    line22.set_dashes([1, 3, 3, 0])

    ax1.set_xlabel('Time steps')
    ax1.legend()

    return fig