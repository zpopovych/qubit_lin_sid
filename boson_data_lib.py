import numpy as np
import h5py
from rho_lib import *

def extract_g(f_name):
    with h5py.File(f_name, 'r') as f:
        g = list(f.keys())
    return g

def extract_y3(f_name, g):
    with h5py.File(f_name, 'r') as f:
        p0 = np.array(f[g]['p0'])  # ground state population
        p1 = np.array(f[g]['p1'])   # exited state population
        assert (np.max(p0 + p1) - np.min(p0 + p1) < 1E-3)

        s_re = np.array(f[g]['s_re'])  # the coherence Real part
        s_im = np.array(f[g]['s_im'])   # the coherence Imag part

        t = np.array(f[g]['t']).reshape(-1)
        t = t.flatten()
        dt = t[1] - t[0]
        assert ((np.abs((t[1:] - t[:-1]) - dt) < 1e-6).all())

    y = np.vstack((p1, s_re, s_im)) # exited state population, coherence Real, Imag parts
    return y.T, dt

def extract_y4(f_name, g):
    with h5py.File(f_name, 'r') as f:
        p0 = np.array(f[g]['p0'])  # ground state population
        p1 = np.array(f[g]['p1'])  # exited state population
        assert (np.max(p0 + p1) - np.min(p0 + p1) < 1E-3)

        s_re = np.array(f[g]['s_re'])  # the coherence Real part
        s_im = np.array(f[g]['s_im'])  # the coherence Imag part

        t = np.array(f[g]['t'])
        t = t.flatten()
        dt = t[1] - t[0]
        assert ((np.abs((t[1:] - t[:-1]) - dt) < 1e-6).all())

    y = np.vstack((p0, p1, s_re, s_im)) # ground, exited states, coherence Real, Imag parts
    return y.T, dt


def extract_rho(f_name, g):
    vec, dt = extract_y3(f_name, g)
    # vec [0] - p1 excited state population
    # vec [1] - coherence Real
    # vec [2] - coherence Imag
    rho = rho_of_vec(vec)
    # rho[:,0,0] excited state population
    # rho[:,1,1] ground state population
    # rho[:,0,1] coherence
    # rho[:,1,0] complex conjugate of coherence
    return rho, dt

def save_model_g(f_mod_name, model, g):
    Ac = model.Ac
    C = model.C
    x0 = model.x0
    with h5py.File(f_mod_name, 'a') as f:
        grp = f.require_group(str(g))
        grp.create_dataset('Ac', Ac.shape, dtype=Ac.dtype, data=Ac)
        grp.create_dataset('C', C.shape, dtype='float64', data=C)
        grp.create_dataset('x0', x0.shape, dtype='float64', data=x0)

def load_model_g(f_mod_name, g):
    with h5py.File(f_mod_name, 'r') as f:
        Ac = np.array(f[str(g)]['Ac'])
        C = np.array(f[str(g)]['C'])
        x0 = np.array(f[str(g)]['x0'])
        mod = linear_model(Ac, C, x0)
    return mod

def load_model_prc_g(f_mod_name, prc, g):
    with h5py.File(f_mod_name, 'r') as f:
        Ac = np.array(f[str(prc)][str(g)]['Ac'])
        C = np.array(f[str(prc)][str(g)]['C'])
        x0 = np.array(f[str(prc)][str(g)]['x0'])
        mod = linear_model(Ac, C, x0)
    return mod

def load_model_g_rank(f_mod_name, g, rank):
    with h5py.File(f_mod_name, 'r') as f:
        Ac = np.array(f[str(g)][str(rank)]['Ac'])
        C = np.array(f[str(g)][str(rank)]['C'])
        x0 = np.array(f[str(g)][str(rank)]['x0'])
        mod = linear_model(Ac, C, x0)
    return mod

def vis_check_rho(file_name):
    g_list = extract_g(file_name)
    plt.figure(figsize=(10, 10))
    plt.title(file_name + ' for different gamma')

    print('Data from:', file_name)

    for g in g_list:
        rho, dt = extract_rho(file_name, g)
        print("gamma = ", g)
        check_rho(rho)
        ''' 
        t = np.linspace(0., dt * rho.shape[0], rho.shape[0]) * float(g)
        plt.plot(t, rho[:, 0, 0].real, 'b-', linewidth=0.8)  # p1 excited state population
        plt.text(t[-1], rho[-1, 0, 0].real, g)
        plt.plot(t, rho[:, 1, 0].real, 'r--')
        plt.text(t[-1], rho[-1, 1, 0].real, g, c='red') 
        '''

    '''
    plt.xlim(0, 50)
    plt.ylim(0, 1)
    plt.plot(0, rho[0, 0, 0].real, c='blue', label='|0> - exited state population')
    plt.plot(0, rho[0, 1, 0].real, 'r--', label='Real part of coherence')
    plt.legend()
    plt.savefig(file_name.replace('.h5', '.pdf'))
    plt.show()
    '''