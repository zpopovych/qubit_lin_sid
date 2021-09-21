import os
import time
from datetime import datetime
from rho_lib import *
from boson_data_lib import *

os.chdir('/home/zah/PycharmProjects/Kurt2021/2021JUN6')
print('Current directory', os.getcwd())

identification_precision = 1e-6

identification_precision_set = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

very_beg = datetime.now()
current_time = very_beg.strftime("%Y%m%d_%H%M_%S")
print(" \n Identification of the two level quantum system")
print("(from density matrix evolution in initial states |0>, |1>, |X>, |Y> )")
print("Began ", very_beg.strftime("%Y %B %d, %H:%M:%S"))

#os.chdir('Dodeca_set_qubit')

f0_name = 'State0_data.h5'
f1_name = 'State1_data.h5'
fX_name = 'StateX_data.h5'
fY_name = 'StateY_data.h5'

f_out_name = "Identified_models_all_100_ranks_"+str(current_time) +".hdf5"
print("Identified models will be saved to file: ", f_out_name)

g0_list = set(extract_g(f0_name))  # State 0
g1_list = set(extract_g(f1_name))  # State 1
gX_list = set(extract_g(fX_name))  # State X
gY_list = set(extract_g(fY_name))  # State Y

g01XY_list  = g0_list & g1_list & gX_list & gY_list
print('Available for all states:', g01XY_list)

print('Extra gammas for initial states:')
print('State0', g0_list ^ g01XY_list)
print('State1', g1_list ^ g01XY_list)
print('StateX', gX_list ^ g01XY_list)
print('StateY', gY_list ^ g01XY_list)



g_set = ['0.25133', '0.79477', '2.5133', '7.9477', '25.133', '79.477', '251.33']

# g_set = ['0.25133', '0.79477', '2.5133', '25.133', '251.33', '7.9477', '79.477']
# g_set = ['25.133', '251.33', '7.9477', '79.477']
# g_set = ['0.25133', '79.477', '25.133', '7.9477', '2.5133', '0.79477']

for g in g_set:
    print('\n Identifying for g=', g)

    y0, dt0 = extract_y3(f0_name, g)  # State 0
    m0 = y0.shape[1]
    y1, dt1 = extract_y3(f1_name, g)  # State 1
    m1 = y1.shape[1]
    yX, dtX = extract_y3(fX_name, g)  # State X
    mX = y1.shape[1]
    yY, dtY = extract_y3(fY_name, g)  # State Y
    mY = yY.shape[1]

    # Stack all three
    assert (dt0 == dt1 == dtX == dtY)
    dt = dt0
    assert (m0 == m1 == mX == mY == 3)
    m = m0 + m1 + mX + mY

    # Choose the smallest lengh of time series
    ser_min = min(y0.shape[0], y1.shape[0], yX.shape[0], yY.shape[0])

    # Stack time series for all 4 initial states
    Y = np.hstack((y0[:ser_min], y1[:ser_min], yX[:ser_min], yY[:ser_min]))
    Y.shape += 1,

    #  The input time series Y is a 3D numpy array:
    # 1st dimension - time
    # 2nd dimension - number of outputs m
    # 3rd dimension = 1 (in field free case)
    print(' Y.shape=', Y.shape)

    #  Run identification function
    func_start = datetime.now()
    H, U, S, m = hankel_and_svd(Y)
    func_run_time = datetime.now() - func_start
    print('Identification time:' + str(func_run_time.seconds) + '.' + str(func_run_time.microseconds) + ' sec.')

    for rank in range(3, 100):
        print(rank, end=" ")
        model = identify_from_hankel_and_svd(H, U, m, rank, dt)
        Ac = model.Ac
        C = model.C
        x0 = model.x0
        #  Save identified model to file
        with h5py.File(f_out_name, 'a') as f:
            g_grp = f.require_group(str(g))
            grp = g_grp.require_group(str(rank))
            grp.create_dataset('Ac', Ac.shape, dtype=Ac.dtype, data=Ac)
            grp.create_dataset('C', C.shape, dtype='float64', data=C)
            grp.create_dataset('x0', x0.shape, dtype='float64', data=x0)
            #grp.create_dataset('Sigma', S.shape, dtype=S.dtype, data=S)
            #grp.create_dataset('H', H.shape, dtype=H.dtype, data=H)

#exit()

tt = datetime.now() - very_beg # Total time
sec = tt.seconds
print('\n Total run time (all g for combined Y):')
print('  '+ str(sec//3600)+":"+str((sec//60)%60)+':'+str(sec%60)+'.'+str(tt.microseconds%1e6))