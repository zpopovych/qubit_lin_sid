import os
import h5py
import numpy as np
from scipy import linalg
from rho_lib import rho_of_y
import matplotlib.pyplot as plt


g_set =  [ '0.25133', '0.79477', '2.5133', '7.9477', '25.133', '79.477', '251.33']
sid_models_file = 'Identified_models_all_100_ranks_20210823_1446_18'
fixed_models_file = sid_models_file + '_fixed'

os.chdir('/home/zah/PycharmProjects/Kurt2021/2021JUN6')
print('Current directory', os.getcwd())

f = h5py.File(fixed_models_file +'.hdf5', 'w')
f.close()


for g in g_set:
    print('g=', g)
    for rank in range(3, 100):
        print('rank=', rank)
        with h5py.File(sid_models_file +'.hdf5', 'r') as f:
            Ac = np.array(f[str(g)][str(rank)]['Ac'])
            C = np.array(f[str(g)][str(rank)]['C'])
            x0 = np.array(f[str(g)][str(rank)]['x0'])

        w, vl, vr = linalg.eig(Ac, left=True, right=True)

        if np.any(w[np.greater(w.real, 1e-3)]): print('Bad model for g = %s, rank = %i' % (g, rank))

        # eliminate positive eigen values
        w[np.greater(w.real, 0)] = w[np.greater(w.real, 0)] - w.real[np.greater(w.real, 0)]
        #w[w.real > 0] = w[w.real > 0].imag * 1j
        s_ind =np.abs(w).argmin()
        y = C @ vr[:, s_ind]
        for rho in rho_of_y(y[np.newaxis, :]):
            rho = rho[0]
            if np.trace(rho @ rho) > 1.1:
                print(w[s_ind])
                print('Trace > 1')

        Ac_fixed = vr @ np.diag(w / np.diag(vl.conj().T @ vr)) @ vl.conj().T

        #with h5py.File(fixed_models_file +'6.hdf5', 'a') as f:
        #    g_grp = f.require_group(str(g))
        #    grp = g_grp.require_group(str(rank))
        #    grp.create_dataset('Ac', Ac.shape, dtype=Ac_fixed.dtype, data=Ac_fixed)
        #    grp.create_dataset('C', C.shape, dtype='float64', data=C)
        #    grp.create_dataset('x0', x0.shape, dtype='float64', data=x0)