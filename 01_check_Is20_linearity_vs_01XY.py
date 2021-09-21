import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import time
from datetime import datetime
from rho_lib import *
from boson_data_lib import *
from qutip import *
from datetime import datetime

is_data = []
prec= 1e-1

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

def bloch_vector(rho):
    n = [2*rho[0,1].real, -2*rho[0,1].imag, rho[0,0].real-rho[1,1].real]
    return n

runs_counter = 0
problem_counter = 0
bad_init_rho=[]

max_error_df = pd.DataFrame(columns=['State', 'Gamma', 'MaxError'])

print('Current directory', os.getcwd())

g0_list = extract_g('State0_data.h5')
g1_list = extract_g('State1_data.h5')
gX_list = extract_g('StateX_data.h5')
gY_list = extract_g('StateY_data.h5')

g01XY_list  = set(g0_list) & set(g1_list) & set(gX_list) & set(gY_list)

print("Gamma present for all base states:", g01XY_list)

for i in range(20):  #range(20):
    f_name = 'Is_' + str(i + 1) + '_data.h5'
    g_list = set(extract_g(f_name)) & set(g01XY_list)

    # g_list = ['0.79477']
    g_set = ['0.25133', '0.79477', '2.5133', '7.9477', '25.133', '79.477', '251.33']

    from scipy.interpolate import PchipInterpolator, interp1d, CubicHermiteSpline

    print(g_list)
    for g in g_list:
        print(g)

        rho_kurt, dt = extract_rho(f_name, g)
        t = np.linspace(start=0.,stop=dt*len(rho_kurt), num=len(rho_kurt))

        kurt_interpol_00_re = PchipInterpolator(t, rho_kurt[:, 0, 0].real)
        #kurt_interpol_00_im = interp1d(t, rho_kurt[:, 0, 0].imag, kind='linear')

        kurt_interpol_01_re = PchipInterpolator(t, rho_kurt[:, 0, 1].real)
        kurt_interpol_01_im = PchipInterpolator(t, rho_kurt[:, 0, 1].imag)

        kurt_interpol_10_re = PchipInterpolator(t, rho_kurt[:, 1, 0].real)
        kurt_interpol_10_im = PchipInterpolator(t, rho_kurt[:, 1, 0].imag)

        kurt_interpol_11_re = PchipInterpolator(t, rho_kurt[:, 1, 1].real)
        #kurt_interpol_11_im = interp1d(t, rho_kurt[:, 1, 1].imag, kind='linear')

        rho0, dt0 = extract_rho('State0_data.h5', g)
        rho1, dt1 = extract_rho('State1_data.h5', g)
        rhoX, dtX = extract_rho('StateX_data.h5', g)
        rhoY, dtY = extract_rho('StateY_data.h5', g)

        assert (dt0 == dt1 == dtX == dtY)
        dt01XY = dt0

        #print('dt01XY=', dt01XY)
        n = min(rho0.shape[0], rho1.shape[0], rhoX.shape[0], rhoY.shape[0])

        if t.max() < dt01XY*n:
            n = int(t.max() // dt01XY)

        t01XY = np.linspace(start=0.,stop=dt01XY*n, num=n)

        #print(t.max(), t01XY.max())


        rho_kurt_interpol = np.empty(shape=(n,2,2), dtype=complex)

        rho_kurt_interpol[:, 0, 0] = kurt_interpol_00_re(t01XY) #+ 0j
        rho_kurt_interpol[:, 0, 1] = kurt_interpol_01_re(t01XY) + 1j * kurt_interpol_01_im(t01XY)
        rho_kurt_interpol[:, 1, 0] = kurt_interpol_10_re(t01XY) + 1j * kurt_interpol_10_im(t01XY)
        rho_kurt_interpol[:, 1, 1] = kurt_interpol_11_re(t01XY) #+ 0j


        rho_comb = rho_01XY(rho_kurt[0], rho0[0:n], rho1[0:n], rhoX[0:n], rhoY[0:n])


        runs_counter += 1

        max_err = np.max(np.abs(rho_kurt_interpol[:n] - rho_comb[:n]))

        data_row = [{'State': i , 'Gamma': float(g), 'MaxError':  max_err}]
        max_error_df = max_error_df.append(data_row, ignore_index=True)

        #if np.max(np.abs(rho_kurt[:n] - rho_comb[:n])) > 1:

        if max_err > prec:
            problem_counter += 1
            #bad_init_rho.append(rho_kurt)

            print('gamma:', g)
            print('state:', i)
            #plot2rho(rho_kurt[:n], rho_comb[:n], 'Density matrix for Initial State #' + str(i + 1) + r', $\gamma =$' + str(g),
            #l1='Dodeca', l2='01XY comb')



print('Number of problematic runs:', problem_counter)
print('Total number of runs:', runs_counter)

print(max_error_df)

print(max_error_df.max(numeric_only=True))


very_beg = datetime.now()
current_time = very_beg.strftime("%Y%m%d_%H%M_%S")
max_error_df.to_pickle("linearity_prec_"+str(current_time) +".pkl")
max_error_df.to_excel("linearity_check_"+str(current_time) +".xlsx")


df = max_error_df.groupby(['Gamma']).max(numeric_only=True).unstack(level=0).reset_index()


print(df)

gammas = df['Gamma'].to_numpy()
lin_err = df.iloc[: , -1].to_numpy()

print(gammas)
print(lin_err)

adjy=np.array([5e-6, 0, -2e-4, 0, -5e-4, -7e-4, 0])
adjx=np.array([-0.1, 0.2, 0, 2, 0, 0, 30])
plt.loglog(gammas, lin_err, '-*')
for i in range(len(gammas)):
    plt.text(gammas[i]+adjx[i], lin_err[i]+adjy[i], np.str(gammas[i]))
plt.title('Linear decomposition error')
plt.xlabel(r'Coupling, $\gamma$')
plt.ylabel('Error')
plt.xlim(1e-1, 1e3)
plt.show()

#print(bad_init_rho)

#rho = [Qobj(_) for _ in bad_init_rho]
#pnts = [bloch_vector(_) for _ in bad_init_rho]
#print(pnts)
#b = Bloch3d()
#b = Bloch()
#b.add_points([[1,0,0], [0,1,0], [0,0,1]])

#for pnt in pnts: b.add_points(pnt)

#b.add_points(pnts)
#b.add_states(rho)
#b.show()
